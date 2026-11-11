"""Network architectures.

A NetSpec describes one network "style": its kind, its layer sizes and the
activation function of each hidden layer. The same forward pass (`_forward`)
is used in two ways:

* TorchNet: one network with gradients, trained by PPO.
* BatchedNet: a whole population of P networks run side by side with no
  gradients. Used by the genetic algorithm, and to test every trained network.

Both use the same flat parameter layout, so a weight vector trained by PPO can
be dropped straight into BatchedNet. tests/test_arena.py checks the two agree.

Kinds
    mlp     plain feed-forward layers
    resmlp  feed-forward, but each layer after the first adds to its input (skip connections)
    rnn     one Elman recurrent layer: h' = act(Wx x + Wh h + b)
    gru     one gated recurrent unit layer
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .activations import torch_act


@dataclass(frozen=True)
class NetSpec:
    kind: str = "mlp"
    hidden: tuple = (64, 64)
    acts: tuple = ("tanh", "tanh")

    def __post_init__(self):
        object.__setattr__(self, "hidden", tuple(self.hidden))
        object.__setattr__(self, "acts", tuple(self.acts))
        if self.kind not in ("mlp", "resmlp", "rnn", "gru"):
            raise ValueError(self.kind)
        if self.kind in ("rnn", "gru") and len(self.hidden) != 1:
            raise ValueError("recurrent nets have exactly one hidden layer")
        if self.kind != "gru" and len(self.acts) != len(self.hidden):
            raise ValueError("need one activation per hidden layer")
        if self.kind == "resmlp" and len(set(self.hidden)) != 1:
            raise ValueError("resmlp layers must all be the same width")

    @property
    def recurrent(self):
        return self.kind in ("rnn", "gru")

    @property
    def state_size(self):
        return self.hidden[0] if self.recurrent else 0

    @property
    def label(self):
        sizes = "x".join(str(h) for h in self.hidden)
        if len(set(self.hidden)) == 1 and len(self.hidden) > 1:
            sizes = f"{len(self.hidden)}x{self.hidden[0]}"
        if self.kind == "gru":
            return f"gru {sizes}"
        acts = self.acts[0] if len(set(self.acts)) == 1 else ">".join(self.acts)
        return f"{self.kind} {sizes} {acts}"

    def to_dict(self):
        return {"kind": self.kind, "hidden": list(self.hidden), "acts": list(self.acts)}

    @staticmethod
    def from_dict(d):
        return NetSpec(d["kind"], tuple(d["hidden"]), tuple(d["acts"]))


def layout(spec, n_in, n_out):
    """Names and shapes of the parameters, in the order they sit in the flat vector.
    Weight matrices are stored as [inputs, outputs]."""
    shapes = []
    if spec.kind in ("mlp", "resmlp"):
        prev = n_in
        for i, h in enumerate(spec.hidden):
            shapes += [(f"W{i}", (prev, h)), (f"b{i}", (h,))]
            prev = h
    elif spec.kind == "rnn":
        H = spec.hidden[0]
        shapes += [("Wx", (n_in, H)), ("Wh", (H, H)), ("b", (H,))]
        prev = H
    else:  # gru
        H = spec.hidden[0]
        shapes += [("Wi", (n_in, 3 * H)), ("Wh", (H, 3 * H)), ("bi", (3 * H,)), ("bh", (3 * H,))]
        prev = H
    shapes += [("Wo", (prev, n_out)), ("bo", (n_out,))]
    return shapes


def count_params(spec, n_in, n_out):
    return sum(int(np.prod(s)) for _, s in layout(spec, n_in, n_out))


def _forward(spec, p, fns, x, h):
    """The forward pass shared by both classes. `p` maps parameter names to tensors:
    weights [in, out] and biases [out] for a single network, or [P, in, out] and
    [P, 1, out] for a population (x is then [P, E, in])."""
    kind = spec.kind
    if kind == "mlp":
        for i, f in enumerate(fns):
            x = f(x @ p[f"W{i}"] + p[f"b{i}"])
    elif kind == "resmlp":
        x = fns[0](x @ p["W0"] + p["b0"])
        for i in range(1, len(fns)):
            x = x + fns[i](x @ p[f"W{i}"] + p[f"b{i}"])
    elif kind == "rnn":
        h = fns[0](x @ p["Wx"] + h @ p["Wh"] + p["b"])
        x = h
    else:  # gru
        H = spec.state_size
        gi = x @ p["Wi"] + p["bi"]
        gh = h @ p["Wh"] + p["bh"]
        r = torch.sigmoid(gi[..., :H] + gh[..., :H])
        z = torch.sigmoid(gi[..., H:2 * H] + gh[..., H:2 * H])
        n = torch.tanh(gi[..., 2 * H:] + r * gh[..., 2 * H:])
        h = (1.0 - z) * n + z * h
        x = h
    return x @ p["Wo"] + p["bo"], h


class BatchedNet:
    """P networks of the same style evaluated together, without gradients.

    Inputs are [P, E, n_in]: E observations for each of the P networks.
    """

    def __init__(self, spec, n_in, n_out):
        self.spec, self.n_in, self.n_out = spec, n_in, n_out
        self.shapes = layout(spec, n_in, n_out)
        self.sizes = [int(np.prod(s)) for _, s in self.shapes]
        self.n_params = sum(self.sizes)
        self.fns = [torch_act(a) for a in spec.acts]
        self.P = 0
        self.p = {}

    def set_params(self, flat):
        flat = torch.as_tensor(np.asarray(flat), dtype=torch.float32)
        if flat.ndim == 1:
            flat = flat[None, :]
        if flat.shape[1] != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {flat.shape[1]}")
        self.P = flat.shape[0]
        off = 0
        for (name, shape), size in zip(self.shapes, self.sizes):
            a = flat[:, off:off + size]
            off += size
            self.p[name] = (a.reshape((self.P,) + shape) if len(shape) == 2
                            else a.reshape(self.P, 1, size)).contiguous()

    def initial_state(self, E):
        if not self.spec.recurrent:
            return None
        return torch.zeros(self.P, E, self.spec.state_size)

    @torch.no_grad()
    def forward(self, x, h=None):
        """x: tensor [P, E, n_in]. Returns (logits [P, E, n_out], new state)."""
        return _forward(self.spec, self.p, self.fns, x, h)


def init_population(spec, n_in, n_out, P, rng):
    """Random starting weights for P networks, plus the per-parameter scale used
    to size mutations (1 / sqrt(fan-in) of the layer the parameter belongs to)."""
    cols, scales = [], []
    fan_in = 1
    for name, shape in layout(spec, n_in, n_out):
        size = int(np.prod(shape))
        if len(shape) == 2:
            fan_in = shape[0]
            s = 1.0 / math.sqrt(fan_in)
            cols.append(rng.normal(0.0, s, (P, size)))
        else:
            s = 1.0 / math.sqrt(fan_in)
            cols.append(np.zeros((P, size)))
        scales.append(np.full(size, s))
    return np.concatenate(cols, axis=1), np.concatenate(scales)


class TorchNet(nn.Module):
    """One trainable network. Hidden layers start orthogonal (gain sqrt 2), which is
    the usual PPO initialisation; the output layer starts small (`out_gain`)."""

    def __init__(self, spec, n_in, n_out, out_gain=0.01):
        super().__init__()
        self.spec, self.n_in, self.n_out = spec, n_in, n_out
        self.names = []
        self.plist = nn.ParameterList()
        for name, shape in layout(spec, n_in, n_out):
            t = torch.zeros(shape)
            if len(shape) == 2:
                if name == "Wo":
                    gain = out_gain
                elif spec.kind == "gru" or name == "Wh":
                    gain = 1.0
                else:
                    gain = math.sqrt(2.0)
                nn.init.orthogonal_(t, gain)
            self.names.append(name)
            self.plist.append(nn.Parameter(t))
        self.fns = [torch_act(a) for a in spec.acts]

    def initial_state(self, n):
        return torch.zeros(n, self.spec.state_size)

    def step(self, x, h=None):
        return _forward(self.spec, dict(zip(self.names, self.plist)), self.fns, x, h)

    def forward_seq(self, x, h0, reset):
        """x [T, N, n_in]; reset [T, N] is 1 where step t starts a new episode."""
        if not self.spec.recurrent:
            out, _ = self.step(x)
            return out, h0
        p = dict(zip(self.names, self.plist))
        outs, h = [], h0
        for t in range(x.shape[0]):
            h = h * (1.0 - reset[t]).unsqueeze(1)
            o, h = _forward(self.spec, p, self.fns, x[t], h)
            outs.append(o)
        return torch.stack(outs), h

    def flat(self):
        return np.concatenate([q.detach().cpu().numpy().ravel() for q in self.plist]).astype(np.float64)


def make_torch_net(spec, n_in, n_out, out_gain=0.01):
    return TorchNet(spec, n_in, n_out, out_gain)
