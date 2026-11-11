"""Make the animated GIFs: the best agent of each training method, side by side on the same scenario.

    python make_gifs.py              # writes results/gifs/<env>.gif and copies the champions
                                     # it used to results/champions/

For each environment it takes the single best run of PPO, of the genetic
algorithm and of NEAT (by test score) and plays all three on one scenario
none of them has seen.
"""
import argparse
import glob
import json
import os
import shutil

from arena.envs import ENV_ORDER, ENVS
from arena.evaluate import load_model, policy_from_model
from arena.render import record_episode, save_gif, side_by_side

METHODS = (("ppo", "PPO"), ("ga", "Genetic algorithm"), ("neat", "NEAT"))
PANEL = (330, 270)
EVERY = {"drone": 2}      # the drone's 25-second episodes are drawn every other step to keep the file small
# scenario shown for each environment (any number works; these were picked to be typical, not flattering)
SCENARIO = {"lander": 5, "memory_lander": 5, "drone": 5, "racer": 5}


def best_runs(out_dir):
    best = {}
    for path in glob.glob(os.path.join(out_dir, "runs", "*.json")):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        key = (r["env"], r["method"])
        if key not in best or r["test_return"] > best[key]["test_return"]:
            best[key] = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=None, help="scenario number (default: one fixed per environment)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "gifs"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "champions"), exist_ok=True)
    best = best_runs(a.out)
    index = {}
    for env in ENV_ORDER:
        columns = []
        seed = a.seed if a.seed is not None else SCENARIO[env]
        for method, name in METHODS:
            r = best.get((env, method))
            if not r:
                continue
            src = os.path.join(a.out, "models", r["id"] + ".npz")
            dst = os.path.join(a.out, "champions", f"{env}__{method}.npz")
            shutil.copyfile(src, dst)
            index[f"{env}__{method}"] = {"run": r["id"], "config": r["label"], "test_return": r["test_return"],
                                         "test_metric": r["test_metric"], "metric_name": r["metric_name"]}
            policy = policy_from_model(load_model(dst))
            label = f"{name}: {r['label'].replace('neat ', '')}"
            frames, ret, metric = record_episode(env, policy, 900_000 + seed, label=label, frame_size=PANEL,
                                                 every=EVERY.get(env, 1))
            columns.append(frames)
            print(f"{env:14s} {method:5s} {r['label']:28s} this episode: return {ret:7.1f}, {r['metric_name']} {metric:.2f}")
        if columns:
            frames = side_by_side(columns)
            path = os.path.join(a.out, "gifs", env + ".gif")
            save_gif(frames, path, ms=ENVS[env].frame_ms * EVERY.get(env, 1), colors=96)
            print(f"   -> {path}  ({len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")
    with open(os.path.join(a.out, "champions", "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
