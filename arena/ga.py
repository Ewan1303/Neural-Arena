"""Genetic algorithm over the weights of a fixed network (no gradients).

Each generation:
  1. every network in the population flies the same few random scenarios;
  2. the best fifth become parents, and the very best few are copied unchanged;
  3. each child is a copy of a random parent, sometimes mixed weight-by-weight
     with a second parent (crossover), plus Gaussian noise on every weight (mutation).

The whole population is simulated at once as one big NumPy batch.
"""
import numpy as np

from .envs import make_env
from .evaluate import EVAL_EPISODES, EVAL_SEED, Curve, cpu_seconds, NetPolicy, run_episodes
from .nets import BatchedNet, init_population

TOP_K = 5   # how many of a generation's best are re-scored at each checkpoint


def train_ga(env_name, spec, seed=0, generations=100, pop=128, episodes=4, sigma=0.1,
             elite_frac=0.05, parent_frac=0.2, crossover=0.3, n_evals=20, progress=None):
    rng = np.random.default_rng(seed)
    env = make_env(env_name, pop * episodes, n_unique=episodes)
    eval_env = make_env(env_name, TOP_K * EVAL_EPISODES, n_unique=EVAL_EPISODES)
    n_in, n_out = env.obs_dim, env.n_actions
    net = BatchedNet(spec, n_in, n_out)
    params, scale = init_population(spec, n_in, n_out, pop, rng)
    n_elite = max(1, int(round(pop * elite_frac)))
    n_parents = max(2, int(round(pop * parent_frac)))
    eval_every = max(1, generations // n_evals)

    curve = Curve()
    best_score, best_params = -np.inf, params[0].copy()
    train_time, steps = 0.0, 0
    for g in range(generations + 1):
        t0 = cpu_seconds()
        policy = NetPolicy(spec, n_in, n_out, params)
        ret, _, used = run_episodes(env, policy, seed * 1_000_003 + g + 1)
        fitness = ret.reshape(pop, episodes).mean(axis=1)
        steps += used
        order = np.argsort(-fitness)
        train_time += cpu_seconds() - t0

        if g % eval_every == 0 or g == generations:
            # Fitness from a handful of episodes is noisy, so the generation's top few are
            # re-scored on the fixed evaluation scenarios and the best of them is recorded.
            top = params[order[:TOP_K]]
            ret, met, _ = run_episodes(eval_env, NetPolicy(spec, n_in, n_out, top), EVAL_SEED)
            ret, met = ret.reshape(len(top), -1), met.reshape(len(top), -1)
            k = int(ret.mean(axis=1).argmax())
            ev = {"return_mean": float(ret[k].mean()), "metric_mean": float(met[k].mean())}
            curve.add(steps, train_time, ev, generation=g, train_best=fitness[order[0]],
                      train_mean=fitness.mean())
            if ev["return_mean"] > best_score:
                best_score, best_params = ev["return_mean"], top[k].copy()
            if progress:
                progress(f"gen {g:4d}  train best {fitness[order[0]]:8.1f}  eval {ev['return_mean']:8.1f}"
                         f"  {env.metric_name} {ev['metric_mean']:.2f}")
        if g == generations:
            break

        t0 = cpu_seconds()
        parents = params[order[:n_parents]]
        n_children = pop - n_elite
        children = parents[rng.integers(0, n_parents, n_children)]
        mates = parents[rng.integers(0, n_parents, n_children)]
        mix = (rng.random(n_children) < crossover)[:, None] & (rng.random(children.shape) < 0.5)
        children = np.where(mix, mates, children)
        children = children + sigma * scale * rng.standard_normal(children.shape)
        params = np.concatenate([params[order[:n_elite]], children], axis=0)
        train_time += cpu_seconds() - t0

    model = {"kind": "net", "spec": spec.to_dict(), "n_in": n_in, "n_out": n_out, "params": best_params}
    return {
        "curve": curve.to_dict(),
        "train_time": train_time,
        "env_steps": steps,
        "n_params": net.n_params,
        "model": model,
        "hyper": {"generations": generations, "pop": pop, "episodes": episodes, "sigma": sigma,
                  "elite_frac": elite_frac, "parent_frac": parent_frac, "crossover": crossover},
    }
