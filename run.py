"""Run the experiments.

    python run.py                      # everything (about 490 training runs)
    python run.py --suite neat         # one suite: activations, architectures, mixed, ga, neat
    python run.py --only lander        # only runs whose id contains this text
    python run.py --scale 0.05         # shrink every budget, for a quick trial
    python run.py --list               # show what would run

Runs that already have a result file are skipped, so it is safe to stop and restart.
"""
import argparse
import os
import random
import sys
import time

# one thread per run: the runs themselves are spread over the CPU cores
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from multiprocessing import Pool

from arena.experiments import build_suite, run_experiment


def _work(args):
    exp, out_dir = args
    try:
        r = run_experiment(exp, out_dir)
        return exp["id"], r["test_return"], r["test_metric"], r["train_time"], None
    except Exception as e:  # keep the rest of the batch going
        import traceback
        return exp["id"], float("nan"), float("nan"), 0.0, f"{e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", default="results")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    exps = build_suite(a.scale, tuple(a.seeds) if a.seeds else (0, 1, 2))
    if a.suite != "all":
        exps = [e for e in exps if e["suite"] == a.suite]
    if a.only:
        exps = [e for e in exps if a.only in e["id"]]
    todo = [e for e in exps if not os.path.exists(os.path.join(a.out, "runs", e["id"] + ".json"))]
    print(f"{len(exps)} experiments selected, {len(exps) - len(todo)} already done, {len(todo)} to run")
    if a.list:
        for e in todo:
            print("  ", e["id"])
        return
    if not todo:
        return
    random.Random(0).shuffle(todo)          # mix long and short runs so the time estimate is steady
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8")
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (eid, ret, metric, secs, err) in enumerate(pool.imap_unordered(_work, [(e, a.out) for e in todo]), 1):
            elapsed = time.time() - t0
            eta = elapsed / k * (len(todo) - k)
            line = (f"[{k:3d}/{len(todo)}] {eid:62s} score {ret:8.1f}  metric {metric:5.2f}  "
                    f"train {secs:6.1f}s  | elapsed {elapsed / 60:5.1f} min, about {eta / 60:5.1f} min left")
            if err:
                line += "\n   FAILED: " + err
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    print(f"done in {(time.time() - t0) / 60:.1f} minutes")


if __name__ == "__main__":
    sys.exit(main())
