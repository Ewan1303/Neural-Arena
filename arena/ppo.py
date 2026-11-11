"""PPO (proximal policy optimisation): the gradient-descent trainer.

A standard actor-critic PPO with clipped updates and generalised advantage
estimation. The actor and the critic are two separate networks built from the
same NetSpec, so whatever architecture and activation functions are being
tested apply to both.

Recurrent networks (rnn, gru) are trained with backpropagation through time
over chunks of `bptt` steps. Each chunk starts from the hidden state the network
actually had at that moment during the rollout, and the state is reset at
episode boundaries.
"""
import numpy as np
import torch

from .envs import make_env
from .evaluate import EVAL_EPISODES, Curve, NetPolicy, cpu_seconds, evaluate
from .nets import count_params, make_torch_net


def train_ppo(env_name, spec, seed=0, total_steps=1_000_000, n_envs=64, rollout=128, epochs=4,
              minibatches=4, lr=1e-3, gamma=0.99, lam=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5,
              max_grad=0.5, bptt=16, n_evals=20, progress=None):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = make_env(env_name, n_envs, seed=seed, autoreset=True)
    eval_env = make_env(env_name, EVAL_EPISODES)
    n_in, n_out = env.obs_dim, env.n_actions
    actor = make_torch_net(spec, n_in, n_out, out_gain=0.01)
    critic = make_torch_net(spec, n_in, 1, out_gain=1.0)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr, eps=1e-5)
    recurrent = spec.recurrent
    T, N = rollout, n_envs
    n_updates = max(1, total_steps // (T * N))
    eval_every = max(1, n_updates // n_evals)
    scale = env.reward_scale

    obs_buf = torch.zeros(T, N, n_in)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    logp_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N)
    reset_buf = torch.zeros(T, N)
    ha_buf = torch.zeros(T, N, spec.state_size)     # hidden states before each step
    hc_buf = torch.zeros(T, N, spec.state_size)
    if recurrent and T % bptt:
        raise ValueError("rollout must be a multiple of bptt")

    curve = Curve()
    best_score, best_params = -np.inf, actor.flat()
    train_time, steps = 0.0, 0

    def checkpoint(u):
        nonlocal best_score, best_params
        flat = actor.flat()
        ev = evaluate(env_name, NetPolicy(spec, n_in, n_out, flat), env=eval_env)
        recent = env.finished_returns[-200:]
        curve.add(steps, train_time, ev, update=u,
                  train_mean=float(np.mean(recent)) if recent else float("nan"))
        if ev["return_mean"] > best_score:
            best_score, best_params = ev["return_mean"], flat
        if progress:
            progress(f"update {u:4d}  steps {steps:8d}  eval {ev['return_mean']:8.1f}"
                     f"  {env.metric_name} {ev['metric_mean']:.2f}")

    obs = env.reset()
    ha, hc = actor.initial_state(N), critic.initial_state(N)
    reset_flag = torch.ones(N)
    checkpoint(0)
    for u in range(1, n_updates + 1):
        t0 = cpu_seconds()
        for group in opt.param_groups:
            group["lr"] = lr * (1.0 - 0.9 * (u - 1) / n_updates)     # linear decay to 10%
        with torch.no_grad():
            for t in range(T):
                o = torch.as_tensor(obs, dtype=torch.float32)
                obs_buf[t] = o
                reset_buf[t] = reset_flag
                if recurrent:
                    ha_buf[t], hc_buf[t] = ha, hc
                    keep = (1.0 - reset_flag).unsqueeze(1)
                    ha, hc = ha * keep, hc * keep
                logits, ha = actor.step(o, ha)
                value, hc = critic.step(o, hc)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                action = dist.sample()
                obs, reward, done, info = env.step(action.numpy())
                r = torch.as_tensor(reward * scale, dtype=torch.float32)
                trunc = info["truncated"]
                if trunc.any():
                    # the episode was cut by the time limit, not ended: add the value of what came next
                    idx = torch.as_tensor(np.flatnonzero(trunc))
                    fo = torch.as_tensor(info["final_obs"][trunc], dtype=torch.float32)
                    v_next, _ = critic.step(fo, hc[idx] if recurrent else None)
                    r[idx] += gamma * v_next.squeeze(1)
                act_buf[t] = action
                logp_buf[t] = dist.log_prob(action)
                val_buf[t] = value.squeeze(1)
                rew_buf[t] = r
                done_buf[t] = torch.as_tensor(done, dtype=torch.float32)
                reset_flag = done_buf[t].clone()
            o = torch.as_tensor(obs, dtype=torch.float32)
            hc_last = hc * (1.0 - reset_flag).unsqueeze(1) if recurrent else hc
            last_value = critic.step(o, hc_last)[0].squeeze(1)
            adv = torch.zeros(T, N)
            gae = torch.zeros(N)
            for t in reversed(range(T)):
                next_value = last_value if t == T - 1 else val_buf[t + 1]
                not_done = 1.0 - done_buf[t]
                delta = rew_buf[t] + gamma * next_value * not_done - val_buf[t]
                gae = delta + gamma * lam * not_done * gae
                adv[t] = gae
            ret = adv + val_buf
        steps += T * N

        n_chunks = T // bptt
        offsets = torch.arange(bptt).unsqueeze(1)               # [bptt, 1]
        for _ in range(epochs):
            if recurrent:
                perm = rng.permutation(n_chunks * N)            # one training item = one chunk of one env
            else:
                perm = rng.permutation(T * N)
            for b in np.array_split(perm, minibatches):
                bi = torch.as_tensor(b)
                if recurrent:
                    start, e = (bi // N) * bptt, bi % N
                    tt = start.unsqueeze(0) + offsets           # [bptt, B] time index of every step
                    ee = e.unsqueeze(0).expand_as(tt)
                    logits, _ = actor.forward_seq(obs_buf[tt, ee], ha_buf[start, e], reset_buf[tt, ee])
                    values, _ = critic.forward_seq(obs_buf[tt, ee], hc_buf[start, e], reset_buf[tt, ee])
                    logits = logits.reshape(-1, n_out)
                    values = values.reshape(-1)
                    sel = lambda x: x[tt, ee].reshape(-1)
                else:
                    logits, _ = actor.step(obs_buf.reshape(T * N, n_in)[bi])
                    values = critic.step(obs_buf.reshape(T * N, n_in)[bi])[0].squeeze(1)
                    sel = lambda x: x.reshape(-1)[bi]
                b_adv = sel(adv)
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                ratio = torch.exp(dist.log_prob(sel(act_buf)) - sel(logp_buf))
                pg_loss = -torch.min(ratio * b_adv, torch.clamp(ratio, 1 - clip, 1 + clip) * b_adv).mean()
                v_loss = 0.5 * (values - sel(ret)).pow(2).mean()
                loss = pg_loss + vf_coef * v_loss - ent_coef * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), max_grad)
                opt.step()
        train_time += cpu_seconds() - t0
        if u % eval_every == 0 or u == n_updates:
            checkpoint(u)

    model = {"kind": "net", "spec": spec.to_dict(), "n_in": n_in, "n_out": n_out, "params": best_params}
    return {
        "curve": curve.to_dict(),
        "train_time": train_time,
        "env_steps": steps,
        "n_params": count_params(spec, n_in, n_out),
        "model": model,
        "hyper": {"total_steps": total_steps, "n_envs": n_envs, "rollout": rollout, "epochs": epochs,
                  "minibatches": minibatches, "lr": lr, "gamma": gamma, "lam": lam, "clip": clip,
                  "ent_coef": ent_coef, "bptt": bptt},
    }
