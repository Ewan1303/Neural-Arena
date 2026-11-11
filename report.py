"""Turn results/runs/*.json into tables and graphs.

    python report.py                 # writes results/figures/*.png, results/summary.csv,
                                     # results/all_runs.csv and RESULTS.md

Scores
    return   the environment's own reward added up over an episode (higher is better)
    metric   the plain-English score: landing rate, waypoints reached, laps completed
    norm     return rescaled per environment so that 0 = random actions and
             1 = the best configuration found in that environment
"""
import argparse
import csv
import glob
import json
import os
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from arena.envs import ENVS, make_env
from arena.evaluate import TEST_EPISODES, TEST_SEED, load_model, run_episodes
from arena.experiments import ACTIVATIONS, ARCHITECTURES, GA_NETS, MIXED, NEAT_VARIANTS
from arena.nets import NetSpec

# ---- look: light surface, quiet axes, colour only on the data
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
METHOD_COLOR = {"ppo": BLUE, "ga": ORANGE, "neat": AQUA}
METHOD_NAME = {"ppo": "PPO (gradient descent)", "ga": "Genetic algorithm", "neat": "NEAT"}
METHOD_SHORT = {"ppo": "PPO", "ga": "GA", "neat": "NEAT"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ENV_ORDER = ["lander", "memory_lander", "drone", "racer"]
ACT_SHORT = {"identity": "linear", "gaussian": "gauss", "sigmoid": "sigm"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 8,
    "axes.labelcolor": INK2, "axes.labelsize": 9.5, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.major.size": 3, "ytick.major.size": 0,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 100, "savefig.dpi": 170,
})


def title_of(env):
    return ENVS[env].title


def metric_of(env):
    return ENVS[env].metric_name


# --------------------------------------------------------------------------- data
def load_runs(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


class RandomPolicy:
    def __init__(self, n_actions, seed=0):
        self.n_actions, self.rng = n_actions, np.random.default_rng(seed)

    def reset(self, n):
        self.n = n

    def act(self, obs):
        return self.rng.integers(0, self.n_actions, self.n)


def random_baselines(out_dir):
    path = os.path.join(out_dir, "baselines.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    base = {}
    for env_name in ENV_ORDER:
        env = make_env(env_name, TEST_EPISODES)
        ret, met, _ = run_episodes(env, RandomPolicy(env.n_actions), TEST_SEED)
        base[env_name] = {"return": float(ret.mean()), "metric": float(met.mean())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=1)
    return base


def aggregate(runs, base):
    """One row per (method, env, configuration), averaged over seeds."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["method"], r["env"], r["label"])].append(r)
    rows = []
    for (method, env, label), rs in groups.items():
        rows.append({
            "method": method, "env": env, "label": label, "seeds": len(rs), "runs": rs,
            "return": float(np.mean([r["test_return"] for r in rs])),
            "return_std": float(np.std([r["test_return"] for r in rs])),
            "metric": float(np.mean([r["test_metric"] for r in rs])),
            "metric_std": float(np.std([r["test_metric"] for r in rs])),
            "metric_seeds": [r["test_metric"] for r in rs],
            "time": float(np.mean([r["train_time"] for r in rs])),
            "steps": float(np.mean([r["env_steps"] for r in rs])),
            "params": float(np.mean([r["n_params"] for r in rs])),
        })
    best = {env: max((r["return"] for r in rows if r["env"] == env), default=1.0) for env in ENV_ORDER}
    for r in rows:
        lo = base[r["env"]]["return"]
        r["norm"] = float(np.clip((r["return"] - lo) / max(best[r["env"]] - lo, 1e-9), 0.0, 1.0))
    return rows


def find(rows, method, env, label):
    for r in rows:
        if r["method"] == method and r["env"] == env and r["label"] == label:
            return r
    return None


def best_of(rows, method, env):
    cands = [r for r in rows if r["method"] == method and r["env"] == env]
    return max(cands, key=lambda r: r["return"]) if cands else None


def fmt_metric(env, v):
    return f"{100 * v:.0f}%" if "rate" in metric_of(env) else f"{v:.2f}"


# --------------------------------------------------------------------------- figure helpers
def save(fig, out_dir, name):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    path = os.path.join(out_dir, "figures", name + ".png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def suptitle(fig, title, subtitle=None, extra=0.0):
    """Title and subtitle above the panels. Positioned in inches, so it sits the
    same distance above the plots whatever the height of the figure. `extra`
    (inches) leaves more room, for example for a legend row."""
    h = fig.get_figheight()
    top = max(ax.get_position().y1 for ax in fig.axes)
    y = top + (0.42 + extra) / h
    lines = subtitle.count("\n") + 1 if subtitle else 0
    if subtitle:
        fig.text(0.0, y, subtitle, fontsize=9.5, color=INK2, ha="left", va="bottom", linespacing=1.35)
    fig.text(0.0, y + (0.2 * lines + 0.07) / h, title, fontsize=13.5, fontweight="bold", color=INK,
             ha="left", va="bottom")


def legend_above(fig, ax, ncol=3):
    """A legend row between the title block and the panel titles."""
    ax_h = ax.get_position().height * fig.get_figheight()
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0 + 0.36 / ax_h), ncol=ncol, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.4, borderaxespad=0)


def hbar_panel(ax, env, labels, rows_by_label, color, show_labels=True, xmax=None):
    """Thin horizontal bars (mean over seeds) with a dot per seed and the value at the tip."""
    y = np.arange(len(labels))[::-1]
    vals = [rows_by_label[l]["metric"] if rows_by_label.get(l) else np.nan for l in labels]
    top = xmax or max(1e-6, np.nanmax([max(rows_by_label[l]["metric_seeds"]) for l in labels if rows_by_label.get(l)]
                                      or [1.0]))
    if "rate" in metric_of(env):
        top = 1.0
    ax.barh(y, np.nan_to_num(vals), height=0.46, color=color, linewidth=0, zorder=2)
    for yi, l in zip(y, labels):
        row = rows_by_label.get(l)
        if not row:
            ax.text(0, yi, "  not run", va="center", ha="left", fontsize=8, color=MUTED)
            continue
        ax.scatter(row["metric_seeds"], [yi] * len(row["metric_seeds"]), s=13, color=INK2, zorder=3,
                   edgecolors=SURFACE, linewidths=0.7)
        ax.text(max(row["metric_seeds"]) + top * 0.035, yi, fmt_metric(env, row["metric"]), va="center",
                ha="left", fontsize=8.3, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels if show_labels else [""] * len(labels))
    ax.set_xlim(0, top * 1.22)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_title(title_of(env))
    ax.set_xlabel(metric_of(env))
    if "rate" in metric_of(env):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def ordered_by_norm(rows, method, labels, envs):
    def score(l):
        vals = [find(rows, method, e, l)["norm"] for e in envs if find(rows, method, e, l)]
        return np.mean(vals) if vals else -1
    return sorted(labels, key=score, reverse=True)


def sweep_figure(rows, out_dir, name, method, labels, envs, title, subtitle, color, pretty=None):
    labels = ordered_by_norm(rows, method, labels, envs)
    pretty = pretty or (lambda l: l)
    fig, axes = plt.subplots(1, len(envs), figsize=(2.2 + 2.75 * len(envs), 0.9 + 0.3 * len(labels)),
                             gridspec_kw={"wspace": 0.12})
    for k, (ax, env) in enumerate(zip(np.atleast_1d(axes), envs)):
        by = {pretty(l): find(rows, method, env, l) for l in labels}
        hbar_panel(ax, env, [pretty(l) for l in labels], by, color, show_labels=(k == 0))
    suptitle(fig, title, subtitle)
    return save(fig, out_dir, name)


# --------------------------------------------------------------------------- figures
def fig_leaderboard(rows, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3), gridspec_kw={"wspace": 0.34})
    for ax, env in zip(axes, ENV_ORDER):
        xs, names = [], []
        for k, method in enumerate(("ppo", "ga", "neat")):
            row = best_of(rows, method, env)
            if not row:
                continue
            ax.bar(k, row["metric"], width=0.34, color=METHOD_COLOR[method], linewidth=0, zorder=2)
            ax.errorbar(k, row["metric"], yerr=row["metric_std"], color=INK2, linewidth=1, capsize=2.5, zorder=3)
            ax.text(k, row["metric"] + row["metric_std"], fmt_metric(env, row["metric"]) + "\n", ha="center",
                    va="bottom", fontsize=9, color=INK, linespacing=0.35)
            xs.append(k)
            names.append(METHOD_SHORT[method] + "\n" + textwrap.fill(row["label"].replace("neat ", ""), 11))
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=7.5, linespacing=1.25)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(0, 1.18 if "rate" in metric_of(env) else max(ax.get_ylim()[1] * 1.16, 1e-6))
        ax.grid(axis="y")
        ax.set_title(title_of(env))
        ax.set_ylabel(metric_of(env))
        if "rate" in metric_of(env):
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    suptitle(fig, "Best configuration of each training method",
             "Score on 200 unseen test scenarios, mean of 3 training seeds (line = spread between seeds). "
             "Under each bar: the network that did best.")
    return save(fig, out_dir, "leaderboard")


def _mean_curve(runs, xkey, grid):
    ys = []
    for r in runs:
        x, y = np.array(r["curve"][xkey], dtype=float), np.array(r["curve"]["eval_return"], dtype=float)
        ys.append(np.interp(grid, x, y, left=y[0], right=np.nan))
    ys = np.array(ys)
    keep = ~np.isnan(ys).any(axis=0)
    return grid[keep], ys[:, keep]


def fig_learning_curves(rows, out_dir, xkey, name, title, subtitle, log):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), gridspec_kw={"wspace": 0.2, "hspace": 0.72})
    for ax, env in zip(axes.ravel(), ENV_ORDER):
        ends = []
        for method in ("ppo", "ga", "neat"):
            row = best_of(rows, method, env)
            if not row:
                continue
            xmax = min(r["curve"][xkey][-1] for r in row["runs"])
            xmin = max(min(x for x in r["curve"][xkey] if x > 0) for r in row["runs"])
            grid = np.geomspace(xmin, xmax, 200) if log else np.linspace(0, xmax, 200)
            g, ys = _mean_curve(row["runs"], xkey, grid)
            ax.fill_between(g, ys.min(axis=0), ys.max(axis=0), color=METHOD_COLOR[method], alpha=0.13, linewidth=0)
            ax.plot(g, ys.mean(axis=0), color=METHOD_COLOR[method], linewidth=2, solid_capstyle="round",
                    label=f"{METHOD_SHORT[method]} · {row['label'].replace('neat ', '')}")
            ax.scatter([g[-1]], [ys.mean(axis=0)[-1]], s=42, color=METHOD_COLOR[method], zorder=4,
                       edgecolors=SURFACE, linewidths=1.6)
            ends.append(ys.mean(axis=0)[-1])
        if log:
            ax.set_xscale("log")
        ax.grid(True)
        ax.set_title(title_of(env))
        ax.set_xlabel("environment steps used (log scale)" if xkey == "steps" else "training time, seconds")
        ax.set_ylabel("score (episode return)")
        if ends:
            ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=3, handlelength=1.2,
                      handletextpad=0.4, columnspacing=1.2, borderaxespad=0, fontsize=8.2)
    suptitle(fig, title, subtitle)
    return save(fig, out_dir, name)


def fig_ga_vs_ppo(rows, out_dir):
    labels = [s.label for s in GA_NETS]
    labels = sorted(labels, key=lambda l: -np.mean([find(rows, "ga", e, l)["norm"] for e in ENV_ORDER
                                                    if find(rows, "ga", e, l)] or [0]))
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 0.9 + 0.5 * len(labels)), gridspec_kw={"wspace": 0.12})
    for k, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        y = np.arange(len(labels))[::-1].astype(float)
        top = 1e-6
        for off, method in ((0.17, "ppo"), (-0.17, "ga")):
            for yi, l in zip(y, labels):
                row = find(rows, method, env, l)
                if not row:
                    continue
                ax.barh(yi + off, row["metric"], height=0.28, color=METHOD_COLOR[method], linewidth=0, zorder=2,
                        label=METHOD_NAME[method] if yi == y[0] else None)
                top = max(top, row["metric"])
        for off, method in ((0.17, "ppo"), (-0.17, "ga")):
            for yi, l in zip(y, labels):
                row = find(rows, method, env, l)
                if row:
                    ax.text(row["metric"] + top * 0.03, yi + off, fmt_metric(env, row["metric"]), va="center",
                            fontsize=7.6, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if k == 0 else [""] * len(labels))
        ax.set_xlim(0, top * 1.22)
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
        ax.set_title(title_of(env))
        ax.set_xlabel(metric_of(env))
        if "rate" in metric_of(env):
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        if "rate" in metric_of(env):
            ax.set_xlim(0, 1.22)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        if k == 0:
            legend_above(fig, ax, ncol=2)
    suptitle(fig, "Same network, two ways of training it",
             "Each network style trained by gradient descent (PPO) and by the genetic algorithm. Mean of 3 seeds.",
             extra=0.3)
    return save(fig, out_dir, "ga_vs_ppo")


def fig_time_vs_score(runs, rows, base, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3), gridspec_kw={"wspace": 0.3})
    for k, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        floor = base[env]["return"]      # runs that did worse than random actions sit on the bottom edge
        for method in ("ppo", "ga", "neat"):
            rs = [r for r in runs if r["env"] == env and r["method"] == method]
            ax.scatter([r["train_time"] for r in rs], [max(r["test_return"], floor) for r in rs], s=26,
                       color=METHOD_COLOR[method], edgecolors=SURFACE, linewidths=0.8, alpha=0.9,
                       label=METHOD_NAME[method], zorder=3)
        ax.grid(True)
        ax.set_title(title_of(env))
        ax.set_xlabel("training time, seconds")
        if k == 0:
            ax.set_ylabel("test score (episode return)")
            legend_above(fig, ax, ncol=3)
    suptitle(fig, "Training time against final score, every run",
             "One dot per training run (all networks, all seeds). Time is training only, on one CPU core. "
             "The bottom of each plot is the score of random actions.", extra=0.3)
    return save(fig, out_dir, "time_vs_score")


def fig_heatmap(rows, out_dir):
    order = []
    for method in ("ppo", "ga", "neat"):
        labels = sorted({r["label"] for r in rows if r["method"] == method},
                        key=lambda l: -np.mean([r["norm"] for r in rows if r["method"] == method and r["label"] == l]))
        order += [(method, l) for l in labels]
    data = np.full((len(order), 4), np.nan)
    fig, ax = plt.subplots(figsize=(6.6, 0.9 + 0.235 * len(order)))
    for i, (method, label) in enumerate(order):
        for j, env in enumerate(ENV_ORDER):
            row = find(rows, method, env, label)
            if row:
                data[i, j] = row["norm"]
    ax.imshow(np.ma.masked_invalid(data), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i, (method, label) in enumerate(order):
        for j, env in enumerate(ENV_ORDER):
            row = find(rows, method, env, label)
            if row:
                ax.text(j, i, fmt_metric(env, row["metric"]), ha="center", va="center", fontsize=7.8,
                        color="#ffffff" if data[i, j] > 0.55 else INK)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{title_of(e)}\n{metric_of(e)}" for e in ENV_ORDER], fontsize=8.3)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{METHOD_SHORT[m]}  ·  {l.replace('neat ', '')}" for m, l in order], fontsize=8.3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    suptitle(fig, "Every configuration on every environment",
             "Number = plain score (mean of 3 seeds). Shade = score relative to the best in that column\n"
             "(white = no better than random actions, dark blue = best). A dash means not run.", extra=0.12)
    return save(fig, out_dir, "heatmap")


def fig_neat(rows, runs, out_dir):
    variants = [v[0] for v in NEAT_VARIANTS] + ["neat tanh recurrent", "neat evolving recurrent"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9), gridspec_kw={"wspace": 0.12})
    for k, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        labels = [v for v in variants if "recurrent" not in v or env == "memory_lander" or k == 0]
        by = {l.replace("neat ", ""): find(rows, "neat", env, l) for l in variants}
        hbar_panel(ax, env, [l.replace("neat ", "") for l in variants], by, AQUA, show_labels=(k == 0))
        for t in ax.texts:
            if t.get_text().strip() == "not run":
                t.set_text("  (memory lander only)")
    suptitle(fig, "NEAT: one fixed activation function, or let evolution choose",
             "'evolving' lets each hidden node mutate its own activation. 'recurrent' allows loops (memory). "
             "Bars: mean of 3 seeds; dots: each seed.")
    p1 = save(fig, out_dir, "neat_variants")

    # which activation functions evolution kept
    from arena.neat import EVOLVABLE
    counts = {a: 0 for a in EVOLVABLE}
    for r in runs:
        if r["method"] == "neat" and "evolving" in r["label"]:
            for a, c in r["extra"].get("champion_activations", {}).items():
                counts[a] = counts.get(a, 0) + c
    total = max(1, sum(counts.values()))
    names = sorted(counts, key=lambda a: -counts[a])
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(names))[::-1]
    share = [counts[a] / total for a in names]
    ax.barh(y, share, height=0.46, color=AQUA, linewidth=0, zorder=2)
    for yi, a, s in zip(y, names, share):
        ax.text(s + 0.006, yi, f"{100 * s:.0f}%  ({counts[a]} nodes)", va="center", fontsize=8.5, color=INK)
    ax.axvline(1 / len(names), color=INK2, linewidth=1, zorder=1)
    ax.text(1 / len(names), len(names) - 0.42, " equal shares", fontsize=8, color=INK2, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(share) * 1.45 + 0.01)
    ax.set_ylim(-0.6, len(names) - 0.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("share of hidden nodes in the winning networks")
    suptitle(fig, "Which activation functions did evolution keep?",
             f"Hidden nodes of every evolving-activation NEAT champion, all environments and seeds ({total} nodes).",
             extra=-0.25)
    p2 = save(fig, out_dir, "neat_activations")

    # how big the networks grew
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.6), gridspec_kw={"wspace": 0.25})
    for k, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        for label, color, name in (("neat tanh", BLUE, "fixed tanh"), ("neat evolving", ORANGE, "evolving activations")):
            row = find(rows, "neat", env, label)
            if not row:
                continue
            gens = [np.array(r["curve"]["generation"]) for r in row["runs"]]
            hid = [np.array(r["curve"]["mean_hidden"]) for r in row["runs"]]
            n = min(len(g) for g in gens)
            mean = np.mean([h[:n] for h in hid], axis=0)
            ax.plot(gens[0][:n], mean, color=color, linewidth=2, label=name, solid_capstyle="round")
            ax.scatter([gens[0][n - 1]], [mean[-1]], s=42, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=4)
        ax.grid(True)
        ax.set_title(title_of(env))
        ax.set_xlabel("generation")
        if k == 0:
            ax.set_ylabel("hidden nodes per network\n(population average)")
            ax.legend(loc="upper left", handlelength=1.4)
    suptitle(fig, "NEAT networks start with no hidden nodes and grow",
             "Average number of hidden nodes across the population, mean of 3 seeds.")
    p3 = save(fig, out_dir, "neat_growth")
    return p1, p2, p3


def draw_genome(ax, genome, env_name, title):
    """Inputs on the left, outputs on the right, hidden nodes placed by depth.
    Blue connections are positive weights, red negative; thicker = stronger."""
    from arena.neat import Genome, node_depths
    env_cls = ENVS[env_name]
    n_in, n_out = env_cls.obs_dim, env_cls.n_actions
    g = Genome.from_dict(genome)
    try:
        depths = node_depths(g, n_in)
    except RecursionError:
        depths = {n: 1 for n in g.nodes}
    hidden = g.hidden_ids(n_in + n_out)
    max_d = max([depths[h] for h in hidden] + [1]) + 1
    pos = {}
    for i in range(n_in):
        pos[i] = (0.0, 1 - (i + 0.5) / n_in)
    for o in range(n_out):
        pos[n_in + o] = (1.0, 1 - (o + 0.5) / n_out)
    by_depth = defaultdict(list)
    for h in hidden:
        by_depth[min(depths[h], max_d - 1)].append(h)
    for d, nodes in by_depth.items():
        for j, h in enumerate(nodes):
            pos[h] = (d / max_d, 1 - (j + 0.5) / len(nodes) + (0.04 if d % 2 else -0.04))
    wmax = max([abs(v[0]) for v in g.conns.values() if v[1]] + [1e-6])
    for (a, b), (w, en) in g.conns.items():
        if not en or a not in pos or b not in pos:
            continue
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=BLUE if w > 0 else RED,
                linewidth=0.4 + 2.2 * abs(w) / wmax, alpha=0.25 + 0.6 * abs(w) / wmax, zorder=1,
                solid_capstyle="round")
    for i in range(n_in):
        ax.scatter(*pos[i], s=46, color=INK2, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        ax.text(pos[i][0] - 0.035, pos[i][1], env_cls.obs_names[i], ha="right", va="center", fontsize=7.6, color=INK2)
    for o in range(n_out):
        ax.scatter(*pos[n_in + o], s=46, color=INK2, zorder=3, edgecolors=SURFACE, linewidths=1.2)
        ax.text(pos[n_in + o][0] + 0.035, pos[n_in + o][1], env_cls.action_names[o], ha="left", va="center",
                fontsize=7.6, color=INK2)
    for h in hidden:
        ax.text(*pos[h], ACT_SHORT.get(g.nodes[h][1], g.nodes[h][1]), ha="center", va="center", fontsize=6.6,
                color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor=AXIS, linewidth=0.8))
    ax.set_xlim(-0.3, 1.36)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")
    ax.set_title(title, fontsize=10)


def fig_neat_networks(rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), gridspec_kw={"wspace": 0.12, "hspace": 0.16})
    for ax, env in zip(axes.ravel(), ENV_ORDER):
        cands = [r for row in rows if row["method"] == "neat" and row["env"] == env and "evolving" in row["label"]
                 and "recurrent" not in row["label"] for r in row["runs"]]
        if not cands:
            ax.axis("off")
            continue
        best = max(cands, key=lambda r: r["test_return"])
        model = load_model(os.path.join(out_dir, "models", best["id"] + ".npz"))
        draw_genome(ax, model["genome"], env,
                    f"{title_of(env)}: {fmt_metric(env, best['test_metric'])} {metric_of(env)}, "
                    f"{best['extra'].get('champion_hidden', 0)} hidden nodes, "
                    f"{best['extra'].get('champion_conns', 0)} connections")
    suptitle(fig, "The networks NEAT evolved (best evolving-activation champion per environment)",
             "Sensors on the left, actions on the right; boxes are hidden nodes labelled with the activation "
             "evolution gave them.\nBlue lines are positive weights, red negative; thicker means stronger.",
             extra=-0.1)
    return save(fig, out_dir, "neat_networks")


# --------------------------------------------------------------------------- tables
def write_csvs(runs, rows, out_dir):
    with open(os.path.join(out_dir, "all_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "method", "env", "config", "seed", "test_return", "test_metric", "metric_name",
                    "train_seconds", "env_steps", "parameters"])
        for r in sorted(runs, key=lambda r: r["id"]):
            w.writerow([r["id"], r["method"], r["env"], r["label"], r["seed"], round(r["test_return"], 2),
                        round(r["test_metric"], 4), r["metric_name"], round(r["train_time"], 1), r["env_steps"],
                        r["n_params"]])
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["env", "method", "config", "seeds", "return_mean", "return_std", "metric_mean", "metric_std",
                    "metric_name", "normalised_score", "train_seconds", "env_steps", "parameters"])
        for r in sorted(rows, key=lambda r: (ENV_ORDER.index(r["env"]), -r["return"])):
            w.writerow([r["env"], r["method"], r["label"], r["seeds"], round(r["return"], 2), round(r["return_std"], 2),
                        round(r["metric"], 4), round(r["metric_std"], 4), metric_of(r["env"]), round(r["norm"], 3),
                        round(r["time"], 1), int(r["steps"]), int(r["params"])])


def md_table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return "\n".join(out)


def steps_text(n):
    return f"{n / 1e6:.1f}M"


def write_markdown(runs, rows, base, out_dir, path="RESULTS.md"):
    L = ["# Full results", "",
         "Generated by `python report.py` from the files in `results/runs/`. Do not edit by hand.", "",
         f"{len(runs)} training runs: {len(rows)} configurations, 3 seeds each. Scores are measured on "
         f"{TEST_EPISODES} test scenarios that were never used during training.", "",
         "- **return**: the environment's reward added up over an episode, mean ± spread between seeds",
         "- **metric**: the plain-English score for that environment",
         "- **norm**: return rescaled so 0 = random actions and 1 = the best configuration in that environment",
         "- **time**: training time in seconds on one CPU core (Intel i5-7400), not counting evaluation",
         "- **steps**: environment steps used in training", ""]
    L += ["## Random-action baseline", "",
          md_table(["environment", "return", "metric"],
                   [[title_of(e), f"{base[e]['return']:.1f}", f"{fmt_metric(e, base[e]['metric'])} {metric_of(e)}"]
                    for e in ENV_ORDER]), ""]
    for env in ENV_ORDER:
        L += [f"## {title_of(env)}", ""]
        lines = []
        for k, r in enumerate(sorted([r for r in rows if r["env"] == env], key=lambda r: -r["return"]), 1):
            lines.append([k, METHOD_SHORT[r["method"]], r["label"].replace("neat ", ""),
                          f"{r['return']:.1f} ± {r['return_std']:.1f}",
                          f"{fmt_metric(env, r['metric'])}", f"{r['norm']:.2f}", f"{r['time']:.0f}",
                          steps_text(r["steps"]), int(r["params"])])
        L += [md_table(["rank", "method", "network", "return", metric_of(env), "norm", "time (s)", "steps",
                        "parameters"], lines), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def print_summary(rows, base):
    for env in ENV_ORDER:
        print(f"\n== {title_of(env)}  (random: {base[env]['return']:.1f}, {fmt_metric(env, base[env]['metric'])})")
        for r in sorted([r for r in rows if r["env"] == env], key=lambda r: -r["return"]):
            print(f"  {METHOD_SHORT[r['method']]:5s} {r['label']:28s} ret {r['return']:7.1f} ±{r['return_std']:5.1f}"
                  f"  {fmt_metric(env, r['metric']):>6s}  norm {r['norm']:.2f}  {r['time']:5.0f}s"
                  f"  {steps_text(r['steps']):>6s}  n={r['seeds']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    runs = load_runs(a.out)
    if not runs:
        raise SystemExit("no results yet: run `python run.py` first")
    base = random_baselines(a.out)
    rows = aggregate(runs, base)
    write_csvs(runs, rows, a.out)
    write_markdown(runs, rows, base, a.out, "RESULTS.md" if a.out == "results" else os.path.join(a.out, "RESULTS.md"))
    if not a.quiet:
        print_summary(rows, base)

    sighted = ["lander", "drone", "racer"]
    fig_leaderboard(rows, a.out)
    fig_learning_curves(rows, a.out, "steps", "learning_curves_steps", "Learning curves by experience used",
                        "Best configuration of each method. Line = mean of 3 seeds, band = lowest to highest seed. "
                        "Scored on 64 fixed evaluation scenarios during training.", log=True)
    fig_learning_curves(rows, a.out, "time", "learning_curves_time", "Learning curves by training time",
                        "Same runs as above, against seconds of training on one CPU core.", log=False)
    sweep_figure(rows, a.out, "activations", "ppo",
                 [NetSpec("mlp", (64, 64), (x, x)).label for x in ACTIVATIONS], sighted,
                 "Activation functions (PPO, two hidden layers of 64)",
                 "Same network and training budget, only the activation function changes. "
                 "Bars: mean of 3 seeds; dots: each seed. Sorted by overall score.", BLUE,
                 pretty=lambda l: l.replace("mlp 2x64 ", ""))
    sweep_figure(rows, a.out, "architectures", "ppo", [s.label for s in ARCHITECTURES], ENV_ORDER,
                 "Network shapes and kinds (PPO, tanh)",
                 "mlp = plain layers, resmlp = with skip connections, rnn and gru = recurrent (they have memory). "
                 "Bars: mean of 3 seeds; dots: each seed.", BLUE)
    sweep_figure(rows, a.out, "mixed", "ppo",
                 [s.label for s in MIXED] + ["mlp 2x64 tanh", "mlp 2x64 relu"], sighted,
                 "A different activation in each layer (PPO)",
                 "'sigmoid>softmax' means sigmoid in the first hidden layer, softmax in the second. "
                 "The two plain networks at the same size are included for reference.", BLUE,
                 pretty=lambda l: l.replace("mlp ", ""))
    fig_ga_vs_ppo(rows, a.out)
    fig_time_vs_score(runs, rows, base, a.out)
    fig_heatmap(rows, a.out)
    fig_neat(rows, runs, a.out)
    fig_neat_networks(rows, a.out)
    print(f"\nwrote figures to {os.path.join(a.out, 'figures')}")


if __name__ == "__main__":
    main()
