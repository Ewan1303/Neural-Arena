"""Make the animated GIFs: the best agent of each training method, side by side on the same scenario.

    python make_gifs.py              # writes results/gifs/<env>.gif and copies the champions
                                     # it used to results/champions/

For each environment it takes the single best run of PPO, of the genetic
algorithm and of NEAT (by test score) and plays all three on one scenario
none of them has seen.

Which scenario? Not a hand-picked one. Each agent plays 20 candidate
scenarios, and the one shown is the scenario in which every agent's result is
closest to its own average, so the GIF is as typical as a single episode can be.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np

from arena.envs import ENV_ORDER, ENVS
from arena.evaluate import load_model, policy_from_model
from arena.render import episode_metric, record_episode, save_gif, side_by_side

METHODS = (("ppo", "PPO"), ("ga", "Genetic algorithm"), ("neat", "NEAT"))
PANEL = (330, 270)
EVERY = {"drone": 2}      # the drone's 25-second episodes are drawn every other step to keep the file small
CANDIDATES = 20
SCENARIO_BASE = 900_000   # far away from the seeds used for training, evaluation and testing


def best_runs(out_dir):
    best = {}
    for path in glob.glob(os.path.join(out_dir, "runs", "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        key = (r["env"], r["method"])
        if key not in best or r["test_return"] > best[key]["test_return"]:
            best[key] = r
    return best


def most_typical_scenario(env, policies):
    """The candidate scenario in which each agent scores closest to its own average."""
    scores = np.array([[episode_metric(env, p, SCENARIO_BASE + s) for s in range(CANDIDATES)] for p in policies])
    span = max(scores.max() - scores.min(), 1e-9)
    gap = np.abs(scores - scores.mean(axis=1, keepdims=True)).sum(axis=0) / span
    return int(gap.argmin()), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--scenario", type=int, default=None, help="force a scenario number instead of the typical one")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "gifs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "champions"), exist_ok=True)
    best = best_runs(a.out)
    index = {}
    for env in ENV_ORDER:
        entries = []
        for method, name in METHODS:
            r = best.get((env, method))
            if not r:
                continue
            src = os.path.join(a.out, "models", r["id"] + ".npz")
            dst = os.path.join(a.out, "champions", f"{env}__{method}.npz")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
            index[f"{env}__{method}"] = {"run": r["id"], "config": r["label"], "test_return": r["test_return"],
                                         "test_metric": r["test_metric"], "metric_name": r["metric_name"]}
            entries.append((method, name, r, policy_from_model(load_model(dst))))
        if not entries:
            continue
        if a.scenario is None:
            scenario, scores = most_typical_scenario(env, [e[3] for e in entries])
            averages = ", ".join(f"{e[0]} {s.mean():.2f}" for e, s in zip(entries, scores))
            print(f"{env}: scenario {scenario} is the most typical of {CANDIDATES} "
                  f"(average {ENVS[env].metric_name} over the candidates: {averages})")
        else:
            scenario = a.scenario
        index[f"{env}__scenario"] = scenario
        columns = []
        for method, name, r, policy in entries:
            label = f"{name}: {r['label'].replace('neat ', '')}"
            frames, ret, metric = record_episode(env, policy, SCENARIO_BASE + scenario, label=label,
                                                 frame_size=PANEL, every=EVERY.get(env, 1))
            columns.append(frames)
            print(f"   {method:5s} {r['label']:24s} this episode: return {ret:7.1f}, {r['metric_name']} {metric:.2f}")
        frames = side_by_side(columns)
        path = os.path.join(a.out, "gifs", env + ".gif")
        save_gif(frames, path, ms=ENVS[env].frame_ms * EVERY.get(env, 1))
        print(f"   -> {path}  ({len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")
    with open(os.path.join(a.out, "champions", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
