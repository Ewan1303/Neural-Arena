"""Base class for the vectorised environments.

Every environment simulates `n` independent copies at once using NumPy arrays,
which is what makes evolving a whole population (hundreds of networks, several
episodes each) fast on a CPU.

Two ways of running:

* autoreset=False (evaluation, GA, NEAT): each copy runs one episode and then
  freezes. Loop until `env.done.all()`.
* autoreset=True (PPO): a copy that finishes is restarted straight away, so the
  batch never empties.

`n_unique` gives common random numbers: only `n_unique` different scenarios are
drawn and they are tiled across the batch. With a population of P networks and
E episodes each (n = P * E, n_unique = E), every network faces the same E
scenarios, so differences in fitness come from the networks and not from luck.
Copy i belongs to network i // E and scenario i % E.
"""
import numpy as np


class VecEnv:
    name = "base"
    title = "Base"
    obs_dim = 0
    n_actions = 0
    max_steps = 400
    reward_scale = 1.0          # PPO multiplies rewards by this; reported scores are unscaled
    metric_name = "metric"      # the human-readable score (landing rate, laps, ...)
    frame_size = (440, 360)     # pixels, for rendering
    frame_ms = 50               # how long one step lasts in a GIF
    static_keys = ()            # per-episode arrays that never change during an episode
    shared_keys = ()            # big static arrays stored once per scenario; index them with self.scn

    def __init__(self, n, seed=0, n_unique=None, autoreset=False):
        self.n = int(n)
        self.n_unique = int(n_unique) if n_unique else self.n
        if self.n % self.n_unique:
            raise ValueError("n must be a multiple of n_unique")
        if autoreset and self.n_unique != self.n:
            raise ValueError("autoreset needs n_unique == n")
        self.autoreset = autoreset
        self.rng = np.random.default_rng(seed)
        self.finished_returns = []
        self.finished_metrics = []
        self.total_steps = 0

    # ---- to implement in a subclass
    def _sample_init(self, m):
        """Return a dict of arrays (first axis m) describing m fresh episodes,
        drawn from self.rng."""
        raise NotImplementedError

    def _dynamics(self, actions):
        """Advance every copy one step. Return (reward, terminated). Must update
        self.metric (the running per-episode score) and self.status itself."""
        raise NotImplementedError

    def _obs(self):
        raise NotImplementedError

    def render_frame(self, i=0, label=None):
        raise NotImplementedError

    # ---- public API
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        init = self._sample_init(self.n_unique)
        reps = self.n // self.n_unique
        self._state_keys = list(init.keys())
        for k, v in init.items():
            if reps > 1 and k not in self.shared_keys:
                v = np.tile(v, (reps,) + (1,) * (v.ndim - 1))
            setattr(self, k, np.array(v, order="C"))
        self.scn = np.arange(self.n) % self.n_unique      # which scenario each copy is running
        self.t = np.zeros(self.n, dtype=np.int64)
        self.done = np.zeros(self.n, dtype=bool)
        self.ep_return = np.zeros(self.n)
        self.metric = np.zeros(self.n)
        self.last_action = np.zeros(self.n, dtype=np.int64)
        self.status = np.zeros(self.n, dtype=np.int64)   # env-specific outcome code, for rendering
        self._trails = {}
        return self._obs()

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        was_done = self.done.copy()
        freeze = (not self.autoreset) and bool(was_done.any())
        if freeze:
            saved = {k: getattr(self, k)[was_done].copy() for k in self._state_keys
                     if k not in self.static_keys and k not in self.shared_keys}
            saved_metric = self.metric[was_done].copy()
            saved_status = self.status[was_done].copy()
            saved_action = self.last_action[was_done].copy()
        self.last_action = actions.copy()
        reward, terminated = self._dynamics(actions)
        if freeze:
            # finished copies stay frozen exactly as they ended
            for k, v in saved.items():
                getattr(self, k)[was_done] = v
            self.metric[was_done] = saved_metric
            self.status[was_done] = saved_status
            self.last_action[was_done] = saved_action
            reward = np.where(was_done, 0.0, reward)
            terminated = terminated & ~was_done
        active = ~was_done
        self.t += active
        truncated = (self.t >= self.max_steps) & ~terminated & active
        done_now = terminated | truncated
        self.ep_return += reward
        self.total_steps += int(active.sum())
        info = {"truncated": truncated}
        if self.autoreset:
            obs = self._obs()
            info["final_obs"] = obs
            if done_now.any():
                self.finished_returns.extend(self.ep_return[done_now].tolist())
                self.finished_metrics.extend(self.metric[done_now].tolist())
                fresh = self._sample_init(int(done_now.sum()))
                for k, v in fresh.items():
                    getattr(self, k)[done_now] = v
                self.t[done_now] = 0
                self.ep_return[done_now] = 0.0
                self.metric[done_now] = 0.0
                self.status[done_now] = 0
                obs = self._obs()
            return obs, reward, done_now, info
        self.done = was_done | done_now
        return self._obs(), reward, self.done.copy(), info
