"""Watch a trained agent live in a window.

    python watch.py lander               # best PPO agent on the lander, new random scenarios forever
    python watch.py drone --method neat  # ppo, ga or neat
    python watch.py racer --model results/models/<run id>.npz
    python watch.py lander --random      # what random actions look like, for comparison

Close the window (or press Esc) to stop. Space starts a new scenario straight away.
"""
import argparse
import os
import tkinter as tk

import numpy as np
from PIL import ImageTk

from arena.envs import ENVS, make_env
from arena.evaluate import load_model, policy_from_model


class RandomPolicy:
    def __init__(self, n_actions):
        self.n_actions, self.rng = n_actions, np.random.default_rng()

    def reset(self, n):
        self.n = n

    def act(self, obs):
        return self.rng.integers(0, self.n_actions, self.n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env", choices=sorted(ENVS))
    ap.add_argument("--method", default="ppo", choices=("ppo", "ga", "neat"))
    ap.add_argument("--model", default=None, help="path to a saved .npz model")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = never)")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    env = make_env(a.env, 1)
    env.frame_size = (660, 540)
    if a.random:
        policy, label = RandomPolicy(env.n_actions), "random actions"
    else:
        path = a.model or os.path.join("results", "champions", f"{a.env}__{a.method}.npz")
        if not os.path.exists(path):
            raise SystemExit(f"no model at {path}. Run `python run.py` and `python make_gifs.py` first, "
                             "or pass --model.")
        model = load_model(path)
        policy = policy_from_model(model)
        label = f"{a.method.upper()}" if not a.model else os.path.basename(path)[:40]

    root = tk.Tk()
    root.title(f"Neural Arena: {ENVS[a.env].title}")
    canvas = tk.Label(root, bd=0)
    canvas.pack()
    state = {"obs": None, "hold": 0, "episodes": 0, "seed": int(np.random.default_rng().integers(1e9)), "img": None}

    def new_episode():
        state["seed"] += 1
        state["obs"] = env.reset(state["seed"])
        policy.reset(1)
        state["hold"] = 0

    def tick():
        if env.done.all():
            state["hold"] += 1
            if state["hold"] > 25:
                state["episodes"] += 1
                if a.episodes and state["episodes"] >= a.episodes:
                    root.destroy()
                    return
                new_episode()
        else:
            state["obs"], _, _, _ = env.step(policy.act(state["obs"]))
        state["img"] = ImageTk.PhotoImage(env.render_frame(0, f"{ENVS[a.env].title}  ·  {label}"))
        canvas.configure(image=state["img"])
        root.after(max(1, int(env.frame_ms / a.speed)), tick)

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("<space>", lambda e: new_episode())
    new_episode()
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
