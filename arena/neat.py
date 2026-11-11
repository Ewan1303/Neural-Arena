"""NEAT (NeuroEvolution of Augmenting Topologies), written from scratch.

Unlike the plain genetic algorithm, NEAT evolves the wiring as well as the
weights. Every network starts with no hidden nodes at all (inputs wired
straight to outputs) and grows: mutations add connections, split a connection
to insert a new node, or remove things again. Networks are grouped into
species by how similar their wiring is, and each species mostly competes with
itself, which gives a new structure time to tune its weights before it has to
beat the established ones.

Two flavours are compared:

* fixed activation: every hidden node uses the same function (e.g. tanh);
* evolving activations: each hidden node carries its own activation function,
  and a mutation can swap it for another from a menu (tanh, sigmoid, relu,
  sin, gaussian, abs, identity).

Output nodes are always linear; the action is whichever output is largest.

The whole population is evaluated in one NumPy batch. The connections of every
network are stored together as flat edge lists, and a feed-forward network is
computed exactly by working through its nodes one depth level at a time.
"""
import math
import random

import numpy as np

from .activations import numpy_act
from .envs import make_env
from .evaluate import EVAL_EPISODES, EVAL_SEED, Curve, cpu_seconds, run_episodes

EVOLVABLE = ("tanh", "sigmoid", "relu", "sin", "gaussian", "abs", "identity")
WEIGHT_LIMIT = 8.0
RECURRENT_ITER = 3    # passes per time step when loops are allowed
TOP_K = 5             # how many of a generation's best are re-scored at each checkpoint

DEFAULTS = dict(
    pop=150, episodes=4,
    p_add_node=0.15, p_add_conn=0.30, p_del_conn=0.05, p_del_node=0.03, p_toggle=0.02,
    p_mutate_weights=0.9, p_weight=0.4, weight_sigma=0.3, p_weight_replace=0.03,
    p_bias=0.3, bias_sigma=0.2, p_activation=0.15,
    p_crossover=0.75, survival=0.25, stagnation=20,
    c_disjoint=2.0, c_weight=0.5, threshold=1.0, threshold_step=0.1, target_species=8,
    max_hidden=40,
)


class Genome:
    """nodes: id -> [bias, activation name] for output and hidden nodes.
    conns: (from, to) -> [weight, enabled]. Inputs are ids 0..n_in-1, outputs follow."""
    __slots__ = ("nodes", "conns", "fitness")

    def __init__(self):
        self.nodes = {}
        self.conns = {}
        self.fitness = -math.inf

    def copy(self):
        g = Genome()
        g.nodes = {k: list(v) for k, v in self.nodes.items()}
        g.conns = {k: list(v) for k, v in self.conns.items()}
        g.fitness = self.fitness
        return g

    def hidden_ids(self, n_io):
        return sorted(k for k in self.nodes if k >= n_io)

    def n_enabled(self):
        return sum(1 for v in self.conns.values() if v[1])

    def to_dict(self):
        return {"nodes": [[k, v[0], v[1]] for k, v in sorted(self.nodes.items())],
                "conns": [[a, b, v[0], bool(v[1])] for (a, b), v in sorted(self.conns.items())]}

    @staticmethod
    def from_dict(d):
        g = Genome()
        g.nodes = {int(k): [float(b), str(a)] for k, b, a in d["nodes"]}
        g.conns = {(int(a), int(b)): [float(w), bool(e)] for a, b, w, e in d["conns"]}
        return g


class Innovation:
    """Hands out node ids so that the same split in two genomes gets the same id,
    which is what lets crossover line genes up."""

    def __init__(self, next_node):
        self.next_node = next_node
        self.split = {}

    def node_for_split(self, key, genome):
        nid = self.split.get(key)
        if nid is None or nid in genome.nodes:
            nid = self.next_node
            self.next_node += 1
            self.split.setdefault(key, nid)
        return nid


def creates_cycle(conns, a, b):
    """Would adding a -> b make a loop?"""
    if a == b:
        return True
    seen = {b}
    while True:
        grew = False
        for i, o in conns:
            if i in seen and o not in seen:
                if o == a:
                    return True
                seen.add(o)
                grew = True
        if not grew:
            return False


def node_depths(genome, n_in):
    """For every output and hidden node: length of the longest path from an input,
    following enabled connections. A node with nothing feeding it has depth 1."""
    incoming = {}
    for (a, b), v in genome.conns.items():
        if v[1]:
            incoming.setdefault(b, []).append(a)
    memo = {}

    def depth(n):
        if n < n_in:
            return 0
        if n in memo:
            return memo[n]
        memo[n] = 1                      # guard, only matters if there is a loop
        memo[n] = 1 + max((depth(a) for a in incoming.get(n, ())), default=0)
        return memo[n]

    return {n: depth(n) for n in genome.nodes}


def genome_depth(genome, n_in):
    return max(node_depths(genome, n_in).values(), default=1)


class NeatPolicy:
    """A batch of genomes acting greedily (also used for a single champion).

    Node values for the whole population live in one array h[E, P * M] (E episodes per
    genome, M node slots per genome). Connections are stored as flat edge lists, so a
    pass is: gather the source values, multiply by the weights, and sum per target.

    Feed-forward genomes are evaluated exactly, one depth level at a time (level 1 is
    every node that only reads inputs, and so on). With loops allowed there is no
    such order, so every node is updated together RECURRENT_ITER times per time step
    and the values carry over to the next step, which is what gives the network memory.
    """

    def __init__(self, genomes, n_in, n_out, recurrent=False):
        self.P, self.I, self.O, self.recurrent = len(genomes), n_in, n_out, recurrent
        n_io = n_in + n_out
        hidden = [g.hidden_ids(n_io) for g in genomes]
        M = self.M = n_io + max((len(h) for h in hidden), default=0)
        bias = np.zeros(self.P * M)
        edges = {}        # level -> list of (target, source, weight)
        acts = {}         # (level, activation) -> list of node indices
        has_input = set()
        nodes_at = {}     # level -> list of node indices
        for p, g in enumerate(genomes):
            slot = {nid: n_io + j for j, nid in enumerate(hidden[p])}
            base = p * M
            depths = None if recurrent else node_depths(g, n_in)
            for nid, (b, act) in g.nodes.items():
                k = base + (nid if nid < n_io else slot[nid])
                bias[k] = b
                level = 1 if recurrent else depths[nid]
                nodes_at.setdefault(level, []).append(k)
                if act != "identity":
                    acts.setdefault((level, act), []).append(k)
            for (a, b), (w, enabled) in g.conns.items():
                if enabled:
                    level = 1 if recurrent else depths[b]
                    kb = base + (b if b < n_io else slot[b])
                    ka = base + (a if a < n_io else slot[a])
                    edges.setdefault(level, []).append((kb, ka, w))
                    has_input.add(kb)
        self.levels = []
        for level in sorted(nodes_at):
            e = sorted(edges.get(level, []))
            tgt = np.array([t for t, _, _ in e], dtype=np.intp)
            src = np.array([s_ for _, s_, _ in e], dtype=np.intp)
            w = np.array([w_ for _, _, w_ in e])
            if len(e):
                starts = np.flatnonzero(np.r_[True, tgt[1:] != tgt[:-1]])
                targets = tgt[starts]
            else:
                starts = targets = np.zeros(0, dtype=np.intp)
            lonely = np.array([k for k in nodes_at[level] if k not in has_input], dtype=np.intp)
            groups = [(numpy_act(name), np.array(sorted(idx), dtype=np.intp))
                      for (lv, name), idx in sorted(acts.items()) if lv == level]
            self.levels.append((src, w, starts, targets, bias[targets], lonely, bias[lonely], groups))
        self.passes = RECURRENT_ITER if recurrent else 1

    def reset(self, n_envs):
        self.E = n_envs // self.P
        self.h = np.zeros((self.E, self.P * self.M))

    def act(self, obs):
        h, I, P, E, M = self.h, self.I, self.P, self.E, self.M
        view = h.reshape(E, P, M)
        view[:, :, :I] = obs.reshape(P, E, I).transpose(1, 0, 2)
        for _ in range(self.passes):
            for src, w, starts, targets, tbias, lonely, lbias, groups in self.levels:
                if len(src):
                    # every node of this level reads the values as they were before the pass
                    sums = np.add.reduceat(h.take(src, axis=1) * w, starts, axis=1)
                    sums += tbias
                    h[:, targets] = sums
                if len(lonely):
                    h[:, lonely] = lbias
                for f, idx in groups:
                    h[:, idx] = f(h.take(idx, axis=1))
                if self.recurrent:
                    np.clip(h, -1e4, 1e4, out=h)
        return view[:, :, I:I + self.O].argmax(axis=2).T.reshape(-1)


# --------------------------------------------------------------------------- genetic operators
def new_genome(n_in, n_out, rnd):
    g = Genome()
    for o in range(n_in, n_in + n_out):
        g.nodes[o] = [0.0, "identity"]
        for i in range(n_in):
            g.conns[(i, o)] = [rnd.gauss(0.0, 1.0), True]
    return g


def mutate(g, n_in, n_out, rnd, innov, cfg, activations, recurrent):
    n_io = n_in + n_out
    hidden = g.hidden_ids(n_io)

    if rnd.random() < cfg["p_add_node"] and len(hidden) < cfg["max_hidden"]:
        enabled = [k for k, v in g.conns.items() if v[1]]
        if enabled:
            key = rnd.choice(enabled)
            w = g.conns[key][0]
            g.conns[key][1] = False
            nid = innov.node_for_split(key, g)
            g.nodes[nid] = [0.0, rnd.choice(activations)]
            g.conns[(key[0], nid)] = [1.0, True]
            g.conns[(nid, key[1])] = [w, True]
            hidden.append(nid)

    if rnd.random() < cfg["p_add_conn"]:
        sources = list(range(n_in)) + hidden + (list(range(n_in, n_io)) if recurrent else [])
        targets = list(range(n_in, n_io)) + hidden
        for _ in range(20):
            a, b = rnd.choice(sources), rnd.choice(targets)
            if (a, b) in g.conns:
                continue
            if not recurrent and creates_cycle(g.conns, a, b):
                continue
            g.conns[(a, b)] = [rnd.gauss(0.0, 1.0), True]
            break

    if rnd.random() < cfg["p_del_conn"] and len(g.conns) > 1:
        del g.conns[rnd.choice(list(g.conns))]

    if rnd.random() < cfg["p_del_node"] and hidden:
        nid = rnd.choice(hidden)
        del g.nodes[nid]
        for key in [k for k in g.conns if nid in k]:
            del g.conns[key]
        hidden.remove(nid)

    if rnd.random() < cfg["p_toggle"] and g.conns:
        v = g.conns[rnd.choice(list(g.conns))]
        v[1] = not v[1]

    if rnd.random() < cfg["p_mutate_weights"]:
        for v in g.conns.values():
            r = rnd.random()
            if r < cfg["p_weight_replace"]:
                v[0] = rnd.gauss(0.0, 1.0)
            elif r < cfg["p_weight_replace"] + cfg["p_weight"]:
                v[0] = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, v[0] + rnd.gauss(0.0, cfg["weight_sigma"])))
        for v in g.nodes.values():
            if rnd.random() < cfg["p_bias"]:
                v[0] = max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, v[0] + rnd.gauss(0.0, cfg["bias_sigma"])))

    if len(activations) > 1 and hidden and rnd.random() < cfg["p_activation"]:
        node = g.nodes[rnd.choice(hidden)]
        node[1] = rnd.choice([a for a in activations if a != node[1]])


def crossover(fit, other, rnd):
    """Child takes its wiring from the fitter parent; where both parents have the
    same gene, either parent's version is used."""
    child = Genome()
    for key, g1 in fit.conns.items():
        g2 = other.conns.get(key)
        if g2 is None:
            child.conns[key] = list(g1)
        else:
            src = g1 if rnd.random() < 0.5 else g2
            enabled = src[1]
            if not (g1[1] and g2[1]):
                enabled = rnd.random() > 0.75
            child.conns[key] = [src[0], enabled]
    for nid, n1 in fit.nodes.items():
        n2 = other.nodes.get(nid)
        child.nodes[nid] = list(n1 if (n2 is None or rnd.random() < 0.5) else n2)
    return child


def distance(g1, g2, cfg):
    c1, c2 = g1.conns, g2.conns
    matching, wdiff = 0, 0.0
    for key, v in c1.items():
        u = c2.get(key)
        if u is not None:
            matching += 1
            wdiff += abs(v[0] - u[0])
    disjoint = len(c1) + len(c2) - 2 * matching
    d = cfg["c_disjoint"] * disjoint / max(len(c1), len(c2), 1)
    if matching:
        d += cfg["c_weight"] * wdiff / matching
    n1, n2 = g1.nodes, g2.nodes
    nmatch, ndiff = 0, 0.0
    for key, v in n1.items():
        u = n2.get(key)
        if u is not None:
            nmatch += 1
            ndiff += abs(v[0] - u[0]) + (1.0 if v[1] != u[1] else 0.0)
    d += cfg["c_disjoint"] * (len(n1) + len(n2) - 2 * nmatch) / max(len(n1), len(n2), 1)
    if nmatch:
        d += cfg["c_weight"] * ndiff / nmatch
    return d


def _allocate(shares, total, minimum):
    """Split `total` offspring between species in proportion to `shares`."""
    k = len(shares)
    minimum = min(minimum, total // k)
    s = sum(shares)
    raw = [minimum + (total - minimum * k) * (x / s if s > 0 else 1.0 / k) for x in shares]
    out = [int(math.floor(r)) for r in raw]
    order = sorted(range(k), key=lambda i: raw[i] - out[i], reverse=True)
    for i in order[: total - sum(out)]:
        out[i] += 1
    return out


def activation_counts(genomes, n_io):
    counts = {}
    for g in genomes:
        for nid, v in g.nodes.items():
            if nid >= n_io:
                counts[v[1]] = counts.get(v[1], 0) + 1
    return counts


def train_neat(env_name, seed=0, generations=100, activations=("tanh",), recurrent=False,
               n_evals=20, progress=None, **overrides):
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    activations = tuple(activations)
    pop, episodes = cfg["pop"], cfg["episodes"]
    rnd = random.Random(seed)
    env = make_env(env_name, pop * episodes, n_unique=episodes)
    eval_env = make_env(env_name, TOP_K * EVAL_EPISODES, n_unique=EVAL_EPISODES)
    n_in, n_out = env.obs_dim, env.n_actions
    n_io = n_in + n_out
    innov = Innovation(n_io)
    genomes = [new_genome(n_in, n_out, rnd) for _ in range(pop)]
    species = []
    threshold = cfg["threshold"]
    eval_every = max(1, generations // n_evals)

    curve = Curve()
    best_score, best_genome = -math.inf, genomes[0].copy()
    train_time, steps = 0.0, 0
    for gen in range(generations + 1):
        t0 = cpu_seconds()
        ret, _, used = run_episodes(env, NeatPolicy(genomes, n_in, n_out, recurrent),
                                    seed * 1_000_003 + gen + 1)
        fit = ret.reshape(pop, episodes).mean(axis=1)
        for g, f in zip(genomes, fit):
            g.fitness = float(f)
        steps += used
        champion = genomes[int(fit.argmax())]
        train_time += cpu_seconds() - t0

        if gen % eval_every == 0 or gen == generations:
            # Fitness from a handful of episodes is noisy, so the generation's top few are
            # re-scored on the fixed evaluation scenarios and the best of them is recorded.
            top = [genomes[i] for i in np.argsort(-fit)[:TOP_K]]
            ret, met, _ = run_episodes(eval_env, NeatPolicy(top, n_in, n_out, recurrent), EVAL_SEED)
            ret, met = ret.reshape(len(top), -1), met.reshape(len(top), -1)
            k = int(ret.mean(axis=1).argmax())
            champion = top[k]
            ev = {"return_mean": float(ret[k].mean()), "metric_mean": float(met[k].mean())}
            hid = [len(g.hidden_ids(n_io)) for g in genomes]
            curve.add(steps, train_time, ev, generation=gen, train_best=float(fit.max()),
                      train_mean=float(fit.mean()), champion_hidden=len(champion.hidden_ids(n_io)),
                      champion_conns=champion.n_enabled(), mean_hidden=float(np.mean(hid)),
                      species=max(len(species), 1), act_counts=activation_counts(genomes, n_io))
            if ev["return_mean"] > best_score:
                best_score, best_genome = ev["return_mean"], champion.copy()
            if progress:
                progress(f"gen {gen:4d}  train best {fit.max():8.1f}  eval {ev['return_mean']:8.1f}"
                         f"  {env.metric_name} {ev['metric_mean']:.2f}  hidden {len(champion.hidden_ids(n_io))}"
                         f"  conns {champion.n_enabled()}  species {max(len(species), 1)}")
        if gen == generations:
            break

        t0 = cpu_seconds()
        # ---- sort the population into species
        for s in species:
            s["members"] = []
        for g in genomes:
            for s in species:
                if distance(g, s["rep"], cfg) < threshold:
                    s["members"].append(g)
                    break
            else:
                species.append({"rep": g, "members": [g], "best": -math.inf, "stale": 0})
        species = [s for s in species if s["members"]]
        if len(species) > cfg["target_species"]:
            threshold += cfg["threshold_step"]
        elif len(species) < cfg["target_species"]:
            threshold = max(0.2, threshold - cfg["threshold_step"])
        for s in species:
            s["members"].sort(key=lambda m: -m.fitness)
            top = s["members"][0].fitness
            if top > s["best"] + 1e-9:
                s["best"], s["stale"] = top, 0
            else:
                s["stale"] += 1
            s["rep"] = s["members"][0]
        species.sort(key=lambda s: -s["members"][0].fitness)
        species = species[:2] + [s for s in species[2:] if s["stale"] < cfg["stagnation"]]

        # ---- share out the next generation and breed
        fmin, fmax = float(fit.min()), float(fit.max())
        span = max(fmax - fmin, 1e-9)
        shares = [sum((m.fitness - fmin) / span for m in s["members"]) / len(s["members"]) + 0.02
                  for s in species]
        spawn = _allocate(shares, pop, 2)
        nxt = []
        for s, n_spawn in zip(species, spawn):
            members = s["members"]
            if n_spawn <= 0:
                continue
            nxt.append(members[0].copy())          # the species champion survives unchanged
            pool = members[:max(1, int(math.ceil(cfg["survival"] * len(members))))]
            if len(pool) < 2 and len(members) >= 2:
                pool = members[:2]
            for _ in range(n_spawn - 1):
                if len(pool) >= 2 and rnd.random() < cfg["p_crossover"]:
                    a, b = rnd.sample(pool, 2)
                    if b.fitness > a.fitness:
                        a, b = b, a
                    child = crossover(a, b, rnd)
                else:
                    child = rnd.choice(pool).copy()
                mutate(child, n_in, n_out, rnd, innov, cfg, activations, recurrent)
                nxt.append(child)
        while len(nxt) < pop:
            child = rnd.choice(species[0]["members"]).copy()
            mutate(child, n_in, n_out, rnd, innov, cfg, activations, recurrent)
            nxt.append(child)
        genomes = nxt[:pop]
        train_time += cpu_seconds() - t0

    hidden = best_genome.hidden_ids(n_io)
    model = {"kind": "neat", "genome": best_genome.to_dict(), "n_in": n_in, "n_out": n_out,
             "recurrent": recurrent, "activations": list(activations)}
    return {
        "curve": curve.to_dict(),
        "train_time": train_time,
        "env_steps": steps,
        "n_params": best_genome.n_enabled() + len(best_genome.nodes),
        "model": model,
        "extra": {"champion_hidden": len(hidden), "champion_conns": best_genome.n_enabled(),
                  "champion_activations": activation_counts([best_genome], n_io)},
        "hyper": {"generations": generations, "activations": list(activations), "recurrent": recurrent,
                  **{k: cfg[k] for k in ("pop", "episodes", "p_add_node", "p_add_conn", "p_activation",
                                         "target_species", "max_hidden")}},
    }
