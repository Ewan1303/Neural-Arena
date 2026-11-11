"""Lander: bring a spacecraft down gently onto a landing pad.

A home-made 2-D take on the classic lunar lander (no physics engine needed).
The craft starts high up at a random position with a random drift and tilt.
It has a main engine that pushes along its own axis and two small side
thrusters that rotate it.

Actions: 0 = nothing, 1 = rotate anticlockwise, 2 = main engine, 3 = rotate clockwise.

MemoryLander is a test of memory. The pad is somewhere different every episode
and the craft only learns where from a blinking beacon: the pad sensor works
for a quarter of a second, then reads zero for three quarters of a second, and
so on. A network with no memory flies blind most of the time, while a recurrent
network can remember the last reading.
"""
import math

import numpy as np

from .base import VecEnv
from .draw import Canvas, rotate

HALF_W = 10.0      # world is x in [-10, 10]
CEILING = 16.0
PAD = 2.0          # half-width of the landing pad
G = 2.0
MAIN = 4.5
SIDE = 3.0         # angular acceleration of a side thruster, rad/s^2
DT = 0.05
SAFE_VY, SAFE_VX, SAFE_ANG = 1.0, 0.7, 0.25
BEACON_PERIOD, BEACON_ON = 20, 5   # MemoryLander: the pad is shown for 5 steps out of every 20

FLYING, LANDED, OFF_PAD, CRASHED, LOST = 0, 1, 2, 3, 4
_STATUS_TEXT = {LANDED: ("LANDED", (120, 230, 140)), OFF_PAD: ("MISSED THE PAD", (240, 200, 90)),
                CRASHED: ("CRASHED", (240, 110, 100)), LOST: ("LOST", (240, 110, 100))}


class Lander(VecEnv):
    name = "lander"
    title = "Lander"
    obs_dim = 6
    n_actions = 4
    max_steps = 400
    reward_scale = 0.02
    metric_name = "landing rate"
    frame_size = (440, 360)
    frame_ms = 50
    static_keys = ("padx",)
    memory = False
    obs_names = ("x", "height", "x speed", "y speed", "tilt", "spin")
    action_names = ("nothing", "turn left", "main engine", "turn right")

    def _sample_init(self, m):
        r = self.rng
        s = {
            "x": r.uniform(-6.0, 6.0, m),
            "y": r.uniform(10.0, 12.0, m),
            "vx": r.uniform(-1.5, 1.5, m),
            "vy": r.uniform(-1.5, 0.0, m),
            "ang": r.uniform(-0.3, 0.3, m),
            "om": r.uniform(-0.3, 0.3, m),
            "padx": r.uniform(-7.0, 7.0, m) if self.memory else np.zeros(m),
        }
        s["phi"] = self._potential(s["x"] - s["padx"], s["y"], s["vx"], s["vy"], s["ang"])
        return s

    @staticmethod
    def _potential(x, y, vx, vy, ang):
        return (-100.0 * np.hypot(x / 10.0, y / 10.0)
                - 100.0 * np.hypot(vx / 5.0, vy / 5.0)
                - 60.0 * np.abs(ang))

    def _dynamics(self, a):
        main = (a == 2).astype(float)
        turn = (a == 1).astype(float) - (a == 3).astype(float)
        self.om = self.om + turn * SIDE * DT
        self.ang = self.ang + self.om * DT
        ax = -np.sin(self.ang) * MAIN * main
        ay = np.cos(self.ang) * MAIN * main - G
        self.vx = self.vx + ax * DT
        self.vy = self.vy + ay * DT
        self.x = self.x + self.vx * DT
        self.y = self.y + self.vy * DT

        touched = self.y <= 0.0
        lost = (np.abs(self.x) > HALF_W) | (self.y > CEILING)
        soft = (np.abs(self.vy) <= SAFE_VY) & (np.abs(self.vx) <= SAFE_VX) & (np.abs(self.ang) <= SAFE_ANG)
        on_pad = np.abs(self.x - self.padx) <= PAD
        landed = touched & soft & on_pad
        off_pad = touched & soft & ~on_pad
        crashed = touched & ~soft
        self.y = np.where(touched, 0.0, self.y)

        phi = self._potential(self.x - self.padx, self.y, self.vx, self.vy, self.ang)
        reward = phi - self.phi - 0.3 * main - 0.03 * np.abs(turn)
        self.phi = phi
        reward = reward + 100.0 * landed + 30.0 * off_pad - 100.0 * (crashed | (lost & ~touched))

        terminated = touched | lost
        self.status = np.where(landed, LANDED, np.where(off_pad, OFF_PAD, np.where(
            crashed, CRASHED, np.where(lost, LOST, self.status))))
        self.metric = np.where(landed, 1.0, self.metric)
        return reward, terminated

    def _obs(self):
        cols = [self.x / 10.0, self.y / 10.0, self.vx / 5.0, self.vy / 5.0, self.ang, self.om / 2.0]
        if self.memory:
            beacon = (self.t % BEACON_PERIOD < BEACON_ON).astype(float)
            cols += [beacon * self.padx / 10.0, beacon]
        return np.stack(cols, axis=1)

    # ---- drawing
    def render_frame(self, i=0, label=None):
        W, H = self.frame_size
        ppu = W / (2 * HALF_W)
        ground = H - 44

        def px(x, y):
            return (W / 2 + x * ppu, ground - y * ppu)

        c = Canvas(W, H, (10, 14, 28))
        stars = np.random.default_rng(7)
        for _ in range(70):
            sx, sy = stars.uniform(0, W), stars.uniform(0, ground - 20)
            b = int(stars.uniform(90, 220))
            c.circle(sx, sy, stars.uniform(0.4, 1.1), fill=(b, b, min(255, b + 30)))
        # hills behind
        hills = [(0, ground)] + [(W * k / 12, ground - 10 - 9 * math.sin(k * 1.7) - 6 * math.sin(k * 0.6 + 1))
                                 for k in range(13)] + [(W, ground)]
        c.poly(hills, fill=(34, 40, 60))
        c.rect(0, ground, W, H, fill=(70, 74, 88))
        for k in range(9):
            cx_ = (k * 53 + 20) % W
            c.d.ellipse([cx_ * 2 - 16, (ground + 14 + (k % 3) * 8) * 2 - 5, cx_ * 2 + 16,
                         (ground + 14 + (k % 3) * 8) * 2 + 5], fill=(58, 62, 76))
        # pad and flags (in MemoryLander they dim whenever the beacon is off)
        padx = float(self.padx[i])
        x0, _ = px(padx - PAD, 0)
        x1, _ = px(padx + PAD, 0)
        beacon_on = (not self.memory) or self.t[i] % BEACON_PERIOD < BEACON_ON
        c.rect(x0, ground - 3, x1, ground + 3, fill=(245, 200, 70) if beacon_on else (120, 112, 84))
        for fx in (x0, x1):
            c.line([(fx, ground), (fx, ground - 26)], fill=(220, 220, 230) if beacon_on else (120, 122, 132),
                   width=1.5)
            c.poly([(fx, ground - 26), (fx + 12, ground - 21), (fx, ground - 16)],
                   fill=(235, 90, 80) if beacon_on else (124, 84, 84))
        if self.memory:
            c.text(W - 8, 8, "beacon ON" if beacon_on else "beacon off", size=10, anchor="ra",
                   fill=(245, 200, 70) if beacon_on else (130, 136, 156))
            if beacon_on:
                for rr in (10, 18, 26):
                    c.d.arc([((x0 + x1) / 2 - rr) * 2, (ground - 8 - rr) * 2, ((x0 + x1) / 2 + rr) * 2,
                             (ground - 8 + rr) * 2], 200, 340, fill=(245, 200, 70, 150), width=2)

        x, y, ang = float(self.x[i]), float(self.y[i]), float(self.ang[i])
        act = int(self.last_action[i]) if (self.status[i] == FLYING and self.t[i] > 0) else 0
        bx, by = px(x, y)
        u = ppu

        def shape(pts):
            # local coordinates in world units, y up, origin at the feet
            out = []
            for lx, ly in pts:
                out.append((bx + lx * u, by - ly * u))
            return rotate(out, -ang, bx, by - 0.55 * u)

        flick = 0.25 * math.sin(self.t[i] * 2.1) + 0.15 * math.sin(self.t[i] * 5.3)
        if act == 2:
            c.poly(shape([(-0.24, 0.3), (0.24, 0.3), (0, -0.85 - flick)]), fill=(255, 150, 40))
            c.poly(shape([(-0.12, 0.3), (0.12, 0.3), (0, -0.4 - flick * 0.5)]), fill=(255, 235, 150))
        if act == 1:   # anticlockwise: puff on the right side pointing down-right
            c.poly(shape([(0.5, 0.75), (0.5, 0.55), (1.05 + flick * 0.3, 0.65)]), fill=(255, 200, 90))
        if act == 3:
            c.poly(shape([(-0.5, 0.75), (-0.5, 0.55), (-1.05 - flick * 0.3, 0.65)]), fill=(255, 200, 90))
        crashed = self.status[i] in (CRASHED,)
        body = (150, 90, 90) if crashed else (214, 218, 232)
        c.line(shape([(-0.35, 0.35), (-0.62, 0.0)]), fill=(170, 174, 190), width=2)
        c.line(shape([(0.35, 0.35), (0.62, 0.0)]), fill=(170, 174, 190), width=2)
        c.line(shape([(-0.78, 0.0), (-0.46, 0.0)]), fill=(170, 174, 190), width=2)
        c.line(shape([(0.78, 0.0), (0.46, 0.0)]), fill=(170, 174, 190), width=2)
        c.poly(shape([(-0.42, 0.3), (0.42, 0.3), (0.52, 0.72), (0.26, 1.08), (-0.26, 1.08), (-0.52, 0.72)]),
               fill=body, outline=(90, 96, 120), width=1)
        c.poly(shape([(-0.18, 0.62), (0.18, 0.62), (0.14, 0.9), (-0.14, 0.9)]), fill=(80, 150, 220))

        speed = math.hypot(self.vx[i], self.vy[i])
        c.text(8, 6, label or self.title, fill=(235, 238, 250), size=13)
        c.text(8, 24, f"speed {speed:4.1f}   height {max(y, 0):4.1f}", fill=(150, 160, 190), size=10)
        st = int(self.status[i])
        if st in _STATUS_TEXT:
            txt, col = _STATUS_TEXT[st]
            c.text(W / 2, 62, txt, fill=col, size=20, anchor="ma")
        elif self.done[i]:
            c.text(W / 2, 62, "OUT OF TIME", fill=(240, 200, 90), size=20, anchor="ma")
        return c.finish()


class MemoryLander(Lander):
    name = "memory_lander"
    title = "Memory Lander"
    obs_dim = 8
    memory = True
    obs_names = Lander.obs_names + ("pad position", "beacon on")
