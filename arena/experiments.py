"""The list of experiments, and the function that runs one of them.

An experiment is a plain dict:
    {"id", "suite", "method" (ppo | ga | neat), "env", "seed", "label",
     "spec" (network style, for ppo and ga) or "neat" (activations, recurrent),
     "budget" (keyword arguments for the trainer)}

Suites
    activations    one architecture (64-64 MLP), twelve activation functions
    architectures  one activation (tanh), eight network shapes and kinds
    mixed          a different activation in each layer (e.g. sigmoid then softmax)
    ga             the genetic algorithm on a selection of the same networks
    neat           NEAT with a fixed activation, and NEAT that evolves its activations
"""
import json
import os
import re
import time

from .envs import ENVS
from .evaluate import TEST_EPISODES, TEST_SEED, evaluate, policy_from_model, save_model
from .neat import EVOLVABLE
from .nets import NetSpec

SEEDS = (0, 1, 2)
ALL_ENVS = ("lander", "memory_lander", "drone", "racer")
SIGHTED = ("lander", "drone", "racer")

ACTIVATIONS = ("relu", "leaky_relu", "elu", "gelu", "silu", "tanh", "sigmoid", "softsign",
               "softplus", "sin", "gaussian", "identity")

ARCHITECTURES = (
    NetSpec("mlp", (16,), ("tanh",)),
    NetSpec("mlp", (64,), ("tanh",)),
    NetSpec("mlp", (256,), ("tanh",)),
    NetSpec("mlp", (64, 64), ("tanh", "tanh")),
    NetSpec("mlp", (64, 64, 64, 64), ("tanh",) * 4),
    NetSpec("resmlp", (64, 64, 64, 64), ("tanh",) * 4),
    NetSpec("rnn", (64,), ("tanh",)),
    NetSpec("gru", (64,), ()),
)

MIXED = (
    NetSpec("mlp", (64, 64), ("tanh", "relu")),
    NetSpec("mlp", (64, 64), ("relu", "tanh")),
    NetSpec("mlp", (64, 64), ("sigmoid", "softmax")),
    NetSpec("mlp", (64, 64), ("relu", "softmax")),
    NetSpec("mlp", (64, 64), ("sin", "tanh")),
    NetSpec("mlp", (64, 64), ("gaussian", "relu")),
    NetSpec("mlp", (64, 64, 64), ("relu", "sigmoid", "tanh")),
)

GA_NETS = (
    NetSpec("mlp", (64, 64), ("tanh", "tanh")),
    NetSpec("mlp", (64, 64), ("relu", "relu")),
    NetSpec("mlp", (64, 64), ("sigmoid", "sigmoid")),
    NetSpec("mlp", (64, 64), ("sin", "sin")),
    NetSpec("mlp", (64, 64), ("sigmoid", "softmax")),
    NetSpec("mlp", (16,), ("tanh",)),
    NetSpec("rnn", (64,), ("tanh",)),
    NetSpec("gru", (64,), ()),
)

NEAT_VARIANTS = (
    ("neat tanh", ("tanh",), False),
    ("neat relu", ("relu",), False),
    ("neat sigmoid", ("sigmoid",), False),
    ("neat evolving", EVOLVABLE, False),
)

# Training budgets. PPO is given a number of environment steps; the evolutionary
# methods a number of generations. They are sized so that one run takes very
# roughly a minute on one core of a desktop CPU, and so that the best
# configurations have clearly stopped improving on the easier tasks.
PPO_STEPS = {"lander": 400_000, "memory_lander": 600_000, "drone": 1_000_000, "racer": 300_000}
GA_GENERATIONS = {"lander": 50, "memory_lander": 60, "drone": 100, "racer": 35}
NEAT_GENERATIONS = {"lander": 100, "memory_lander": 100, "drone": 100, "racer": 50}
PPO_HYPER = dict(n_envs=64, rollout=128, epochs=4, minibatches=8, lr=2e-3, gamma=0.997, lam=0.97)


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _exp(suite, method, env, seed, label, budget, spec=None, neat=None):
    e = {"id": f"{method}__{env}__{_slug(label)}__s{seed}", "suite": suite, "method": method,
         "env": env, "seed": seed, "label": label, "budget": budget}
    if spec is not None:
        e["spec"] = spec.to_dict()
    if neat is not None:
        e["neat"] = neat
    return e


def build_suite(scale=1.0, seeds=SEEDS):
    """Every experiment. `scale` shrinks the budgets (for a quick trial run)."""
    exps = {}

    def add(e):
        exps.setdefault(e["id"], e)       # the 64-64 tanh MLP is in several suites; run it once

    def ppo_budget(env):
        return dict(PPO_HYPER, total_steps=max(16_384, int(PPO_STEPS[env] * scale)))

    def gens(table, env):
        return max(4, int(table[env] * scale))

    for seed in seeds:
        for env in SIGHTED:
            for act in ACTIVATIONS:
                spec = NetSpec("mlp", (64, 64), (act, act))
                add(_exp("activations", "ppo", env, seed, spec.label, ppo_budget(env), spec=spec))
            for spec in MIXED:
                add(_exp("mixed", "ppo", env, seed, spec.label, ppo_budget(env), spec=spec))
        for env in ALL_ENVS:
            for spec in ARCHITECTURES:
                add(_exp("architectures", "ppo", env, seed, spec.label, ppo_budget(env), spec=spec))
            for spec in GA_NETS:
                add(_exp("ga", "ga", env, seed, spec.label,
                         dict(generations=gens(GA_GENERATIONS, env)), spec=spec))
            for label, acts, rec in NEAT_VARIANTS:
                add(_exp("neat", "neat", env, seed, label,
                         dict(generations=gens(NEAT_GENERATIONS, env)),
                         neat={"activations": list(acts), "recurrent": rec}))
        # loops give NEAT memory, which only the memory lander needs
        for label, acts in (("neat tanh recurrent", ("tanh",)), ("neat evolving recurrent", EVOLVABLE)):
            add(_exp("neat", "neat", "memory_lander", seed, label,
                     dict(generations=gens(NEAT_GENERATIONS, "memory_lander")),
                     neat={"activations": list(acts), "recurrent": True}))
    return list(exps.values())


def run_experiment(exp, out_dir="results", progress=None):
    """Train, test on the held-out scenarios, and write results/runs/<id>.json
    plus results/models/<id>.npz. Returns the result dict."""
    os.makedirs(os.path.join(out_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    t0 = time.time()
    method, env, seed = exp["method"], exp["env"], exp["seed"]
    if method == "ppo":
        from .ppo import train_ppo
        out = train_ppo(env, NetSpec.from_dict(exp["spec"]), seed=seed, progress=progress, **exp["budget"])
    elif method == "ga":
        from .ga import train_ga
        out = train_ga(env, NetSpec.from_dict(exp["spec"]), seed=seed, progress=progress, **exp["budget"])
    elif method == "neat":
        from .neat import train_neat
        out = train_neat(env, seed=seed, progress=progress, activations=exp["neat"]["activations"],
                         recurrent=exp["neat"]["recurrent"], **exp["budget"])
    else:
        raise ValueError(method)

    model = out.pop("model")
    test = evaluate(env, policy_from_model(model), episodes=TEST_EPISODES, seed=TEST_SEED)
    result = dict(exp)
    result.update({
        "metric_name": ENVS[env].metric_name,
        "test_return": test["return_mean"],
        "test_return_std": test["return_std"],
        "test_metric": test["metric_mean"],
        "best_eval_return": max(out["curve"]["eval_return"]),
        "train_time": out["train_time"],
        "wall_time": time.time() - t0,
        "env_steps": out["env_steps"],
        "n_params": out["n_params"],
        "hyper": out["hyper"],
        "extra": out.get("extra", {}),
        "curve": out["curve"],
    })
    save_model(os.path.join(out_dir, "models", exp["id"] + ".npz"), model)
    with open(os.path.join(out_dir, "runs", exp["id"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result
