"""Drone: fly a two-rotor drone through as many waypoints as possible in 25 seconds.

Harder than the lander because it never ends in a single manoeuvre: as soon as
one waypoint is reached the next one lights up somewhere else, so the network
has to keep its speed under control while already lining up for the next
target. It sees the current waypoint and the one after it.

Actions: 0 = both rotors off, 1 = left rotor, 2 = right rotor, 3 = both rotors.
Leaving the arena ends the episode.
"""
import math

import numpy as np

from .base import VecEnv
from .draw import Canvas, rotate

HALF_W = 10.0
TOP = 16.0
G = 2.0
ROTOR = 2.3          # acceleration from one rotor
TORQUE = 3.5         # angular acceleration from one rotor on its own
ANG_DAMP = 2.0
LIN_DRAG = 0.15
DT = 0.05
REACH = 1.0          # how close counts as reaching a waypoint
N_WAYPOINTS = 40

FLYING, CRASHED = 0, 1


class Drone(VecEnv):
    name = "drone"
    title = "Drone"
    obs_dim = 11
    n_actions = 4
    max_steps = 500
    reward_scale = 0.05
    metric_name = "waypoints reached"
    frame_size = (440, 360)
    frame_ms = 50
    static_keys = ("wp",)
    obs_names = ("x", "height", "x speed", "y speed", "sin tilt", "cos tilt", "spin",
                 "target dx", "target dy", "next dx", "next dy")
    action_names = ("rotors off", "left rotor", "right rotor", "both rotors")

    def _sample_init(self, m):
        r = self.rng
        s = {
            "x": r.uniform(-3.0, 3.0, m),
            "y": r.uniform(6.0, 10.0, m),
            "vx": np.zeros(m),
            "vy": np.zeros(m),
            "ang": r.uniform(-0.2, 0.2, m),
            "om": np.zeros(m),
            "wp": np.stack([r.uniform(-8.0, 8.0, (m, N_WAYPOINTS)),
                            r.uniform(2.5, 13.5, (m, N_WAYPOINTS))], axis=2),
            "k": np.zeros(m, dtype=np.int64),
        }
        s["dist"] = np.hypot(s["wp"][:, 0, 0] - s["x"], s["wp"][:, 0, 1] - s["y"])
        return s

    def _target(self, offset=0):
        idx = (self.k + offset) % N_WAYPOINTS
        return self.wp[np.arange(len(self.k)), idx]

    def _dynamics(self, a):
        left = ((a == 1) | (a == 3)).astype(float)
        right = ((a == 2) | (a == 3)).astype(float)
        thrust = (left + right) * ROTOR
        # the left rotor on its own lifts the left side, which turns the drone clockwise
        self.om = self.om + ((right - left) * TORQUE - ANG_DAMP * self.om) * DT
        self.ang = self.ang + self.om * DT
        ax = -np.sin(self.ang) * thrust - LIN_DRAG * self.vx
        ay = np.cos(self.ang) * thrust - G - LIN_DRAG * self.vy
        self.vx = self.vx + ax * DT
        self.vy = self.vy + ay * DT
        self.x = self.x + self.vx * DT
        self.y = self.y + self.vy * DT

        tgt = self._target()
        d = np.hypot(tgt[:, 0] - self.x, tgt[:, 1] - self.y)
        reward = self.dist - d
        reached = d < REACH
        reward = reward + 10.0 * reached
        self.k = self.k + reached
        self.metric = self.metric + reached
        if reached.any():
            tgt = self._target()
            d = np.hypot(tgt[:, 0] - self.x, tgt[:, 1] - self.y)
        self.dist = d

        crashed = (np.abs(self.x) > HALF_W) | (self.y < 0.0) | (self.y > TOP)
        reward = reward - 20.0 * crashed
        self.status = np.where(crashed, CRASHED, self.status)
        return reward, crashed

    def _obs(self):
        t1 = self._target(0)
        t2 = self._target(1)
        return np.stack([
            self.x / 10.0, (self.y - 8.0) / 8.0, self.vx / 5.0, self.vy / 5.0,
            np.sin(self.ang), np.cos(self.ang), self.om / 3.0,
            (t1[:, 0] - self.x) / 10.0, (t1[:, 1] - self.y) / 10.0,
            (t2[:, 0] - self.x) / 10.0, (t2[:, 1] - self.y) / 10.0,
        ], axis=1)

    # ---- drawing
    def render_frame(self, i=0, label=None):
        W, H = self.frame_size
        ppu = W / (2 * HALF_W)
        base = H - 4

        def px(x, y):
            return (W / 2 + x * ppu, base - y * ppu)

        c = Canvas(W, H, (16, 22, 34))
        for gx in range(-8, 9, 2):
            c.line([px(gx, 0), px(gx, TOP)], fill=(28, 36, 52), width=1)
        for gy in range(2, 16, 2):
            c.line([px(-HALF_W, gy), px(HALF_W, gy)], fill=(28, 36, 52), width=1)
        c.rect(1, base - TOP * ppu, W - 2, base, outline=(90, 104, 136), width=2)

        x, y, ang = float(self.x[i]), float(self.y[i]), float(self.ang[i])
        trail = self._trails.setdefault(i, [])
        if self.status[i] == FLYING and not self.done[i]:
            trail.append((x, y))
        pts = trail[-70:]
        for j in range(1, len(pts)):
            alpha = int(160 * j / len(pts))
            c.line([px(*pts[j - 1]), px(*pts[j])], fill=(90, 190, 255, alpha), width=1.5)

        k = int(self.k[i])
        nxt = self.wp[i, (k + 1) % N_WAYPOINTS]
        cur = self.wp[i, k % N_WAYPOINTS]
        nx, ny = px(*nxt)
        c.circle(nx, ny, REACH * ppu * 0.8, outline=(120, 110, 70), width=1.5)
        cx_, cy_ = px(*cur)
        pulse = 1.0 + 0.12 * math.sin(self.t[i] * 0.5)
        c.circle(cx_, cy_, REACH * ppu * pulse, fill=(255, 210, 70, 60), outline=(255, 210, 70), width=2)
        c.circle(cx_, cy_, 3, fill=(255, 225, 120))

        bx, by = px(x, y)
        u = ppu

        def shape(pts_):
            return rotate([(bx + lx * u, by - ly * u) for lx, ly in pts_], -ang, bx, by)

        act = int(self.last_action[i]) if (self.status[i] == FLYING and self.t[i] > 0 and not self.done[i]) else 0
        flick = 0.15 * math.sin(self.t[i] * 2.3)
        if act in (1, 3):
            c.poly(shape([(-0.95, -0.08), (-0.65, -0.08), (-0.8, -0.75 - flick)]), fill=(120, 210, 255, 200))
        if act in (2, 3):
            c.poly(shape([(0.65, -0.08), (0.95, -0.08), (0.8, -0.75 - flick)]), fill=(120, 210, 255, 200))
        crashed = self.status[i] == CRASHED
        col = (170, 90, 90) if crashed else (230, 234, 245)
        c.line(shape([(-0.8, 0.0), (0.8, 0.0)]), fill=col, width=3)
        c.poly(shape([(-0.28, -0.16), (0.28, -0.16), (0.2, 0.22), (-0.2, 0.22)]), fill=(255, 140, 60))
        c.line(shape([(-1.15, 0.14), (-0.45, 0.14)]), fill=(150, 160, 185), width=2)
        c.line(shape([(0.45, 0.14), (1.15, 0.14)]), fill=(150, 160, 185), width=2)

        c.text(8, 6, label or self.title, fill=(235, 238, 250), size=13)
        c.text(8, 24, f"waypoints {int(self.metric[i])}   time {self.t[i] * DT:4.1f}s",
               fill=(150, 160, 190), size=10)
        if crashed:
            c.text(W / 2, 62, "CRASHED", fill=(240, 110, 100), size=20, anchor="ma")
        elif self.done[i]:
            c.text(W / 2, 62, f"{int(self.metric[i])} WAYPOINTS", fill=(120, 230, 140), size=20, anchor="ma")
        return c.finish()
