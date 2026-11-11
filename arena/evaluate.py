"""Running policies in the environments, and saving / loading trained ones.

A "policy" here is anything with reset(n_envs) and act(obs) -> actions.
Every method (PPO, GA, NEAT) is scored the same way: greedy actions (the
highest output wins) on fixed sets of scenarios.

    EVAL_SEED  64 scenarios used during training to draw learning curves and to
               pick which checkpoint to keep
    TEST_SEED  200 different scenarios used once at the end for the reported score
"""
import json

import numpy as np
import torch

from .envs import make_env
from .nets import BatchedNet, NetSpec

EVAL_SEED = 10_001
TEST_SEED = 20_002
EVAL_EPISODES = 64
TEST_EPISODES = 200


class NetPolicy:
    """One or many fixed-architecture networks acting greedily."""

    def __init__(self, spec, n_in, n_out, flat):
        self.net = BatchedNet(spec, n_in, n_out)
        self.net.set_params(flat)
        self.P = self.net.P

    def reset(self, n_envs):
        self.E = n_envs // self.P
        self.h = self.net.initial_state(self.E)

    def act(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(self.P, self.E, -1)
        logits, self.h = self.net.forward(x, self.h)
        return logits.argmax(dim=2).reshape(-1).numpy()


def run_episodes(env, policy, seed):
    """Run every copy of env for one episode. Returns (returns, metrics, env steps used)."""
    before = env.total_steps
    obs = env.reset(seed)
    policy.reset(env.n)
    while not env.done.all():
        obs, _, _, _ = env.step(policy.act(obs))
    return env.ep_return.copy(), env.metric.copy(), env.total_steps - before


def evaluate(env_name, policy, episodes=EVAL_EPISODES, seed=EVAL_SEED, env=None):
    env = env or make_env(env_name, episodes)
    ret, met, _ = run_episodes(env, policy, seed)
    return {
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "metric_mean": float(met.mean()),
        "metric_std": float(met.std()),
    }


# --------------------------------------------------------------------------- saving
def save_model(path, model):
    """model is a dict: {"kind": "net", "spec": {...}, "n_in", "n_out", "params": array}
    or {"kind": "neat", "genome": {...}, "n_in", "n_out", ...}. Stored as .npz without pickle."""
    meta = {k: v for k, v in model.items() if k != "params"}
    params = np.asarray(model.get("params", np.zeros(0)), dtype=np.float32)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), params=params)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        model = json.loads(str(z["meta"]))
        model["params"] = z["params"].astype(np.float64)
    return model


def policy_from_model(model):
    if model["kind"] == "net":
        return NetPolicy(NetSpec.from_dict(model["spec"]), model["n_in"], model["n_out"], model["params"])
    from .neat import Genome, NeatPolicy
    return NeatPolicy([Genome.from_dict(model["genome"])], model["n_in"], model["n_out"],
                      recurrent=model.get("recurrent", False))


class Curve:
    """Learning curve: one row per checkpoint."""

    def __init__(self):
        self.rows = []

    def add(self, steps, seconds, ev, **extra):
        row = {"steps": int(steps), "time": float(seconds),
               "eval_return": ev["return_mean"], "eval_metric": ev["metric_mean"]}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                    for k, v in extra.items()})
        self.rows.append(row)

    def to_dict(self):
        keys = sorted({k for r in self.rows for k in r})
        return {k: [r.get(k) for r in self.rows] for k in keys}
