"""The test environments. Use make_env("lander", n=...) to build one."""
from .base import VecEnv
from .drone import Drone
from .lander import MemoryLander, Lander
from .racer import Racer

ENVS = {cls.name: cls for cls in (Lander, MemoryLander, Drone, Racer)}
ENV_ORDER = ["lander", "memory_lander", "drone", "racer"]


def make_env(name, n, seed=0, n_unique=None, autoreset=False):
    return ENVS[name](n, seed=seed, n_unique=n_unique, autoreset=autoreset)


__all__ = ["VecEnv", "ENVS", "ENV_ORDER", "make_env", "Lander", "MemoryLander", "Drone", "Racer"]
