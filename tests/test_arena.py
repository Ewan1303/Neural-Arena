"""Checks that the building blocks behave. Run with:  python -m pytest -q"""
import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.activations import NUMPY, numpy_act, torch_act
from arena.envs import ENV_ORDER, make_env
from arena.evaluate import NetPolicy, load_model, policy_from_model, run_episodes, save_model
from arena.nets import BatchedNet, NetSpec, count_params, init_population, make_torch_net
from arena import neat

SPECS = [
    NetSpec("mlp", (16,), ("relu",)),
    NetSpec("mlp", (12, 10), ("sigmoid", "softmax")),
    NetSpec("mlp", (8, 8, 8), ("gelu", "sin", "gaussian")),
    NetSpec("resmlp", (16, 16, 16), ("tanh", "elu", "silu")),
    NetSpec("rnn", (12,), ("tanh",)),
    NetSpec("gru", (12,), ()),
]


# ----------------------------------------------------------------------------- activations
@pytest.mark.parametrize("name", sorted(NUMPY))
def test_activation_numpy_matches_torch(name):
    import torch
    x = np.linspace(-6, 6, 41).reshape(1, 41)
    a = numpy_act(name)(x)
    b = torch_act(name)(torch.tensor(x)).numpy()
    assert np.allclose(a, b, atol=1e-6), name


# ----------------------------------------------------------------------------- networks
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.label)
def test_population_net_matches_trainable_net(spec):
    """A weight vector taken out of the PPO network gives the same outputs in BatchedNet."""
    import torch
    torch.manual_seed(0)
    n_in, n_out, E, T = 5, 3, 4, 6
    tnet = make_torch_net(spec, n_in, n_out, out_gain=1.0)
    with torch.no_grad():
        for p in tnet.parameters():      # biases start at zero, so randomise everything
            p.add_(0.3 * torch.randn_like(p))
    flat = tnet.flat()
    assert flat.shape == (count_params(spec, n_in, n_out),)
    other = flat + 1.0                   # a second, different network in the same batch
    bnet = BatchedNet(spec, n_in, n_out)
    bnet.set_params(np.stack([other, flat]))
    xs = torch.randn(T, E, n_in)
    h_b = bnet.initial_state(E)
    h_t = tnet.initial_state(E)
    for t in range(T):
        out_b, h_b = bnet.forward(torch.stack([xs[t], xs[t]]), h_b)
        with torch.no_grad():
            out_t, h_t = tnet.step(xs[t], h_t)
        assert torch.allclose(out_b[1], out_t, atol=1e-5)
        assert not torch.allclose(out_b[0], out_t, atol=1e-3)
    # running the whole sequence at once gives the same answer as stepping
    with torch.no_grad():
        seq, _ = tnet.forward_seq(xs, tnet.initial_state(E), torch.zeros(T, E))
    assert torch.allclose(seq[-1], out_t, atol=1e-5)


def test_mutation_scale_matches_layout():
    spec = NetSpec("mlp", (8, 8), ("tanh", "relu"))
    params, scale = init_population(spec, 4, 3, 5, np.random.default_rng(1))
    assert params.shape == (5, count_params(spec, 4, 3)) and scale.shape == (params.shape[1],)
    assert np.isclose(scale[0], 0.5) and np.isclose(scale[-1], 1 / np.sqrt(8))


# ----------------------------------------------------------------------------- environments
@pytest.mark.parametrize("name", ENV_ORDER)
def test_env_shapes_and_determinism(name):
    def rollout():
        env = make_env(name, 6)
        obs = env.reset(123)
        assert obs.shape == (6, env.obs_dim)
        rng = np.random.default_rng(5)
        total = np.zeros(6)
        for _ in range(env.max_steps + 5):
            obs, r, d, _ = env.step(rng.integers(0, env.n_actions, 6))
            assert np.isfinite(obs).all() and np.isfinite(r).all()
            total += r
            if d.all():
                break
        assert env.done.all()
        assert np.allclose(total, env.ep_return)
        return env.ep_return.copy(), env.metric.copy()

    a, b = rollout(), rollout()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("name", ENV_ORDER)
def test_same_network_gets_same_score_anywhere_in_the_population(name):
    """Scenarios are tiled, so P copies of one network must all score the same."""
    P, E = 5, 3
    env = make_env(name, P * E, n_unique=E)
    spec = NetSpec("mlp", (8,), ("tanh",))
    rng = np.random.default_rng(0)
    one, _ = init_population(spec, env.obs_dim, env.n_actions, 1, rng)
    ret, met, _ = run_episodes(env, NetPolicy(spec, env.obs_dim, env.n_actions, np.repeat(one, P, axis=0)), 7)
    ret = ret.reshape(P, E)
    assert np.allclose(ret, ret[0])
    # and the same scenarios again from a plain (untiled) environment
    env1 = make_env(name, E)
    ret1, _, _ = run_episodes(env1, NetPolicy(spec, env.obs_dim, env.n_actions, one), 7)
    assert np.allclose(ret1, ret[0])


@pytest.mark.parametrize("name", ENV_ORDER)
def test_autoreset_keeps_running(name):
    env = make_env(name, 8, seed=3, autoreset=True)
    obs = env.reset()
    rng = np.random.default_rng(1)
    for _ in range(env.max_steps * 2):
        obs, r, d, info = env.step(rng.integers(0, env.n_actions, 8))
        assert np.isfinite(obs).all()
        assert info["final_obs"].shape == obs.shape
    assert len(env.finished_returns) >= 8


@pytest.mark.parametrize("name", ENV_ORDER)
def test_render(name):
    env = make_env(name, 2)
    env.reset(1)
    env.step(np.array([1, 2]))
    img = env.render_frame(0, label="test")
    assert img.size == env.frame_size


def test_lander_can_be_landed_by_a_simple_hand_written_rule():
    """Sanity check that the task is solvable: a small hand-tuned controller lands most of the time."""
    env = make_env("lander", 200)
    env.reset(4)
    while not env.done.all():
        target_ang = np.clip(0.25 * env.x / 10 * 4 + 0.35 * env.vx, -0.5, 0.5)
        turn = (target_ang - env.ang) * 2.0 - env.om * 1.2
        target_vy = -0.25 - 0.22 * env.y
        a = np.where(env.vy < target_vy, 2, np.where(turn > 0.08, 1, np.where(turn < -0.08, 3, 0)))
        env.step(a)
    assert env.metric.mean() > 0.6


# ----------------------------------------------------------------------------- NEAT
def _slow_feedforward(genome, n_in, n_out, x):
    """Reference implementation: evaluate each node recursively."""
    memo = {}

    def value(n):
        if n < n_in:
            return x[n]
        if n in memo:
            return memo[n]
        bias, act = genome.nodes[n]
        total = bias + sum(w * value(a) for (a, b), (w, en) in genome.conns.items() if b == n and en)
        memo[n] = float(numpy_act(act)(np.array(total)))
        return memo[n]

    return np.array([value(o) for o in range(n_in, n_in + n_out)])


def _random_genomes(n, n_in, n_out, rnd, recurrent=False, steps=40):
    innov = neat.Innovation(n_in + n_out)
    cfg = dict(neat.DEFAULTS, p_add_node=0.5, p_add_conn=0.6)
    out = []
    for _ in range(n):
        g = neat.new_genome(n_in, n_out, rnd)
        for _ in range(steps):
            neat.mutate(g, n_in, n_out, rnd, innov, cfg, neat.EVOLVABLE, recurrent)
        out.append(g)
    return out


def test_neat_batch_matches_reference():
    rnd = random.Random(0)
    n_in, n_out = 4, 3
    genomes = _random_genomes(12, n_in, n_out, rnd)
    assert max(len(g.hidden_ids(n_in + n_out)) for g in genomes) >= 3
    policy = neat.NeatPolicy(genomes, n_in, n_out)
    E = 5
    policy.reset(len(genomes) * E)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(len(genomes) * E, n_in))
    actions = policy.act(obs).reshape(len(genomes), E)
    outputs = policy.h.reshape(E, len(genomes), policy.M)[:, :, n_in:n_in + n_out]
    for p, g in enumerate(genomes):
        for e in range(E):
            ref = _slow_feedforward(g, n_in, n_out, obs[p * E + e])
            assert np.allclose(outputs[e, p], ref, atol=1e-8)
            assert actions[p, e] == ref.argmax()
    # a second call must not be affected by what the first one left behind
    again = policy.act(obs).reshape(len(genomes), E)
    assert np.array_equal(actions, again)


def test_neat_feedforward_genomes_have_no_loops():
    rnd = random.Random(1)
    for g in _random_genomes(20, 3, 2, rnd):
        for (a, b) in g.conns:
            assert b >= 3, "nothing may feed into an input"
            assert not (3 <= a < 5), "outputs feed nothing in feed-forward mode"
        without = dict(g.conns)
        for key in g.conns:
            rest = {k: v for k, v in without.items() if k != key}
            assert not neat.creates_cycle(rest, *key)


def test_neat_crossover_and_distance():
    rnd = random.Random(2)
    a, b = _random_genomes(2, 4, 2, rnd)
    cfg = neat.DEFAULTS
    assert neat.distance(a, a, cfg) == 0
    assert neat.distance(a, b, cfg) > 0
    assert math.isclose(neat.distance(a, b, cfg), neat.distance(b, a, cfg))
    child = neat.crossover(a, b, rnd)
    assert set(child.conns) == set(a.conns) and set(child.nodes) == set(a.nodes)


def test_neat_recurrent_runs_and_model_roundtrip(tmp_path):
    rnd = random.Random(3)
    g = _random_genomes(1, 8, 4, rnd, recurrent=True)[0]
    model = {"kind": "neat", "genome": g.to_dict(), "n_in": 8, "n_out": 4, "recurrent": True}
    path = tmp_path / "m.npz"
    save_model(path, model)
    policy = policy_from_model(load_model(path))
    env = make_env("memory_lander", 4)
    ret, _, _ = run_episodes(env, policy, 9)
    ret2, _, _ = run_episodes(env, neat.NeatPolicy([g], 8, 4, recurrent=True), 9)
    assert np.allclose(ret, ret2)


def test_net_model_roundtrip(tmp_path):
    spec = NetSpec("gru", (8,), ())
    rng = np.random.default_rng(0)
    params, _ = init_population(spec, 6, 4, 1, rng)
    model = {"kind": "net", "spec": spec.to_dict(), "n_in": 6, "n_out": 4, "params": params[0]}
    save_model(tmp_path / "n.npz", model)
    policy = policy_from_model(load_model(tmp_path / "n.npz"))
    env = make_env("lander", 4)
    ret, _, _ = run_episodes(env, policy, 9)
    ret2, _, _ = run_episodes(env, NetPolicy(spec, 6, 4, params[0]), 9)
    assert np.allclose(ret, ret2)


# ----------------------------------------------------------------------------- trainers (smoke tests)
def test_ga_improves_on_lander():
    from arena.ga import train_ga
    r = train_ga("lander", NetSpec("mlp", (16,), ("tanh",)), seed=0, generations=12, pop=48, episodes=2, n_evals=2)
    assert r["curve"]["train_best"][-1] > r["curve"]["train_best"][0]
    assert r["env_steps"] > 0 and r["model"]["params"].shape == (r["n_params"],)


def test_neat_runs_with_evolving_activations():
    r = neat.train_neat("lander", seed=0, generations=6, activations=neat.EVOLVABLE, pop=40, episodes=2, n_evals=2)
    assert len(r["curve"]["steps"]) >= 2
    assert r["model"]["kind"] == "neat"


@pytest.mark.parametrize("spec", [NetSpec("mlp", (16, 16), ("sigmoid", "softmax")), NetSpec("gru", (16,), ())],
                         ids=lambda s: s.label)
def test_ppo_runs(spec):
    from arena.ppo import train_ppo
    r = train_ppo("drone", spec, seed=0, total_steps=4096, n_envs=8, rollout=64, n_evals=2)
    assert r["env_steps"] == 4096
    assert np.isfinite(r["curve"]["eval_return"]).all()


def test_memory_lander_beacon_blinks():
    env = make_env("memory_lander", 4)
    obs = env.reset(3)
    assert len(set(np.round(env.padx, 3))) == 4, "pads are in different places"
    seen = []
    for _ in range(45):
        on = obs[:, 7]
        assert np.all((on == 0) | (on == 1)) and len(set(on)) == 1
        if on[0]:
            assert np.allclose(obs[:, 6], env.padx / 10.0)
        else:
            assert np.allclose(obs[:, 6], 0.0), "with the beacon off the pad sensor must read zero"
        seen.append(int(on[0]))
        obs, _, _, _ = env.step(np.full(4, 2))
    assert seen[:20] == [1] * 5 + [0] * 15 and seen[20:25] == [1] * 5
