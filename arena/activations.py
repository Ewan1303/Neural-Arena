"""Activation functions, each in a NumPy and a PyTorch version.

The NumPy versions are used by the evolutionary methods (genetic algorithm and
NEAT), which never need gradients. The PyTorch versions are used by PPO.
Both versions of a function compute the same thing, which tests/test_nets.py checks.
"""
import math

import numpy as np

_GELU_C = math.sqrt(2.0 / math.pi)


def _np_softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


NUMPY = {
    "identity": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
    "gelu": lambda x: 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x * x * x))),
    "silu": lambda x: x * (0.5 * (1.0 + np.tanh(0.5 * x))),
    "tanh": np.tanh,
    "sigmoid": lambda x: 0.5 * (1.0 + np.tanh(0.5 * x)),
    "softsign": lambda x: x / (1.0 + np.abs(x)),
    "softplus": lambda x: np.logaddexp(0.0, x),
    "sin": np.sin,
    "gaussian": lambda x: np.exp(-np.minimum(x * x, 50.0)),
    "abs": np.abs,
    "softmax": _np_softmax,  # acts on a whole layer, not on one unit
}

# Activations that need the whole layer at once (cannot be a per-node function in NEAT).
LAYERWISE = {"softmax"}


def numpy_act(name):
    return NUMPY[name]


def torch_act(name):
    import torch
    import torch.nn.functional as F

    table = {
        "identity": lambda x: x,
        "relu": F.relu,
        "leaky_relu": lambda x: F.leaky_relu(x, 0.01),
        "elu": F.elu,
        "gelu": lambda x: F.gelu(x, approximate="tanh"),
        "silu": F.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softsign": F.softsign,
        "softplus": F.softplus,
        "sin": torch.sin,
        "gaussian": lambda x: torch.exp(-torch.clamp(x * x, max=50.0)),
        "abs": torch.abs,
        "softmax": lambda x: torch.softmax(x, dim=-1),
    }
    return table[name]
