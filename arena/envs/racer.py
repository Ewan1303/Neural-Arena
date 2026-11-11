"""Racer: drive a car around a race track it has never seen, using only distance sensors.

Every episode draws a new random closed track. The car sees nine "lidar" rays
(how far the track edge is in nine directions) and its own speed, nothing else.
It has 30 seconds; the score is how many laps it completes. Touching the edge
of the track ends the episode, and the car has limited grip, so it cannot take
corners flat out and has to brake.

Actions: steering (right, straight, left) x pedal (accelerate, brake) = 6 actions.
action % 3 picks the steering, action // 3 picks the pedal.
"""
import math

import numpy as np

from .base import VecEnv
from .draw import Canvas, rotate

R0 = 30.0                  # average track radius, metres
HALF_WIDTH = 4.0
HARMONICS = np.array([2, 3, 4, 5])
AMPLITUDE = np.array([0.18, 0.14, 0.09, 0.06])
NBINS = 256
EXTENT = 54.0              # the lidar grid covers [-EXTENT, EXTENT] in both directions
GRID = 144
CELL = 2 * EXTENT / GRID
ACCEL, BRAKE, DRAG, ROLL = 9.0, 14.0, 0.014, 0.3
GRIP = 10.0                # maximum sideways acceleration
MAX_YAW = 2.2
DT = 0.1
SUBSTEPS = 2
RAY_ANGLES = np.radians([-90, -55, -30, -12, 0, 12, 30, 55, 90])
RAY_DIST = np.array([1.5, 3, 4.5, 6, 8, 10, 12.5, 15, 18, 22, 27, 33, 40.0])
RAY_MAX = 44.0

DRIVING, CRASHED = 0, 1

# polar coordinates of every grid cell centre, used to rasterise a track quickly
_gc = (np.arange(GRID) + 0.5) * CELL - EXTENT
_GX, _GY = np.meshgrid(_gc, _gc)                       # row index = y, column index = x
_CELL_RHO = np.hypot(_GX, _GY).ravel()
_CELL_BIN = (np.floor((np.arctan2(_GY, _GX) % (2 * np.pi)) / (2 * np.pi) * NBINS).astype(np.int64) % NBINS).ravel()
_THETA = (np.arange(NBINS) + 0.5) * 2 * np.pi / NBINS


class Racer(VecEnv):
    name = "racer"
    title = "Racer"
    obs_dim = 10
    n_actions = 6
    max_steps = 300
    reward_scale = 0.1
    metric_name = "laps completed"
    frame_size = (440, 360)
    frame_ms = 100
    static_keys = ("rtab", "th_start")
    shared_keys = ("grid",)
    obs_names = ("ray -90", "ray -55", "ray -30", "ray -12", "ray 0", "ray +12", "ray +30", "ray +55",
                 "ray +90", "speed")
    action_names = ("right + gas", "straight + gas", "left + gas",
                    "right + brake", "straight + brake", "left + brake")

    def _sample_init(self, m):
        r = self.rng
        amp = r.uniform(0.0, 1.0, (m, len(HARMONICS))) * AMPLITUDE
        phase = r.uniform(0.0, 2 * np.pi, (m, len(HARMONICS)))
        ang = _THETA[None, :, None] * HARMONICS[None, None, :] + phase[:, None, :]
        rtab = R0 * (1.0 + (amp[:, None, :] * np.sin(ang)).sum(axis=2))          # [m, NBINS]
        grid = np.ascontiguousarray(np.abs(_CELL_RHO[None, :] - rtab[:, _CELL_BIN]) < HALF_WIDTH)
        grid = grid.reshape(m, GRID, GRID)

        th0 = r.uniform(0.0, 2 * np.pi, m)
        b = (np.floor(th0 / (2 * np.pi) * NBINS).astype(np.int64)) % NBINS
        rows = np.arange(m)
        rc = rtab[rows, b]
        drdth = (rtab[rows, (b + 1) % NBINS] - rtab[rows, (b - 1) % NBINS]) / (2 * (2 * np.pi / NBINS))
        rho = rc + r.uniform(-1.5, 1.5, m)
        # direction of travel (anticlockwise) along the centre line
        tx = drdth * np.cos(th0) - rc * np.sin(th0)
        ty = drdth * np.sin(th0) + rc * np.cos(th0)
        return {
            "px": rho * np.cos(th0),
            "py": rho * np.sin(th0),
            "psi": np.arctan2(ty, tx) + r.uniform(-0.15, 0.15, m),
            "v": r.uniform(0.0, 5.0, m),
            "theta": th0.copy(),
            "progress": np.zeros(m),
            "rtab": rtab,
            "grid": grid,
            "th_start": th0.copy(),
        }

    def _centre_radius(self, theta):
        b = (np.floor((theta % (2 * np.pi)) / (2 * np.pi) * NBINS).astype(np.int64)) % NBINS
        return self.rtab[np.arange(len(theta)), b]

    def _dynamics(self, a):
        steer = (a % 3 - 1).astype(float)
        gas = (a // 3 == 0)
        crashed = np.zeros(len(a), dtype=bool)
        h = DT / SUBSTEPS
        for _ in range(SUBSTEPS):
            acc = np.where(gas, ACCEL, -BRAKE) - DRAG * self.v * self.v - ROLL
            self.v = np.maximum(self.v + acc * h, 0.0)
            yaw = steer * np.minimum(MAX_YAW, GRIP / np.maximum(self.v, 1e-3))
            self.psi = self.psi + yaw * h
            self.px = self.px + self.v * np.cos(self.psi) * h
            self.py = self.py + self.v * np.sin(self.psi) * h
            rho = np.hypot(self.px, self.py)
            th = np.arctan2(self.py, self.px)
            crashed |= np.abs(rho - self._centre_radius(th)) > HALF_WIDTH
        dth = (th - self.theta + np.pi) % (2 * np.pi) - np.pi
        self.theta = th % (2 * np.pi)
        self.progress = self.progress + dth
        reward = dth * R0 * 0.1 - 10.0 * crashed
        self.metric = np.maximum(self.progress, 0.0) / (2 * np.pi)
        self.status = np.where(crashed, CRASHED, self.status)
        return reward, crashed

    def _lidar(self):
        """Distance to the track edge along each ray, found by sampling points along the ray
        and looking them up in the rasterised track. Work arrays are reused between calls
        because allocating them fresh every step was the slowest part of the environment."""
        n = len(self.px)
        w = getattr(self, "_work", None)
        if w is None or w[0].shape[0] != n:
            shape = (n, len(RAY_ANGLES), len(RAY_DIST))
            w = self._work = (np.empty(shape), np.empty(shape), np.empty(shape, dtype=np.intp),
                              np.empty(shape, dtype=np.intp))
        fx, fy, ix, iy = w
        dirs = self.psi[:, None] + RAY_ANGLES[None, :]                       # [n, rays]
        np.multiply(np.cos(dirs)[:, :, None], RAY_DIST / CELL, out=fx)       # [n, rays, K]
        fx += ((self.px + EXTENT) / CELL)[:, None, None]
        np.multiply(np.sin(dirs)[:, :, None], RAY_DIST / CELL, out=fy)
        fy += ((self.py + EXTENT) / CELL)[:, None, None]
        # anything outside the grid lands on a border cell, and border cells are never track
        np.clip(fx, 0, GRID - 1, out=fx)
        np.clip(fy, 0, GRID - 1, out=fy)
        ix[...] = fx
        iy[...] = fy
        iy *= GRID
        iy += ix
        iy += (self.scn * (GRID * GRID))[:, None, None]
        off = ~self.grid.reshape(-1).take(iy)
        first = off.argmax(axis=2)
        any_off = off.any(axis=2)
        prev = np.where(first > 0, RAY_DIST[np.maximum(first - 1, 0)], 0.0)
        hit = 0.5 * (prev + RAY_DIST[first])
        return np.where(any_off, hit, RAY_MAX)

    def _obs(self):
        lidar = self._lidar() / RAY_MAX
        return np.concatenate([lidar, (self.v / 25.0)[:, None]], axis=1)

    # ---- drawing
    def render_frame(self, i=0, label=None):
        W, H = self.frame_size
        ppu = min(W, H - 30) / (2 * 50.0)
        cx0, cy0 = W / 2, (H + 26) / 2

        def px(x, y):
            return (cx0 + x * ppu, cy0 - y * ppu)

        c = Canvas(W, H, (38, 84, 56))
        rt = self.rtab[i]
        outer = [px((r + HALF_WIDTH) * math.cos(t), (r + HALF_WIDTH) * math.sin(t)) for r, t in zip(rt, _THETA)]
        inner = [px((r - HALF_WIDTH) * math.cos(t), (r - HALF_WIDTH) * math.sin(t)) for r, t in zip(rt, _THETA)]
        c.poly(outer, fill=(64, 66, 74))
        c.poly(inner, fill=(38, 84, 56))
        c.line(outer + outer[:1], fill=(225, 225, 225), width=1.5)
        c.line(inner + inner[:1], fill=(225, 225, 225), width=1.5)
        mid = [px(r * math.cos(t), r * math.sin(t)) for r, t in zip(rt, _THETA)]
        for j in range(0, NBINS, 8):
            c.line([mid[j], mid[(j + 3) % NBINS]], fill=(110, 112, 120), width=1)
        ts = float(self.th_start[i])
        rs = float(self._centre_radius(self.th_start)[i])
        c.line([px((rs - HALF_WIDTH) * math.cos(ts), (rs - HALF_WIDTH) * math.sin(ts)),
                px((rs + HALF_WIDTH) * math.cos(ts), (rs + HALF_WIDTH) * math.sin(ts))],
               fill=(255, 255, 255), width=3)

        x, y, psi = float(self.px[i]), float(self.py[i]), float(self.psi[i])
        trail = self._trails.setdefault(i, [])
        if self.status[i] == DRIVING and not self.done[i]:
            trail.append((x, y))
        pts = trail[-45:]
        for j in range(1, len(pts)):
            c.line([px(*pts[j - 1]), px(*pts[j])], fill=(255, 190, 80, int(170 * j / len(pts))), width=1.5)

        crashed = self.status[i] == CRASHED
        if not crashed and not self.done[i]:
            d = self._lidar()[i]
            for ang, dist in zip(RAY_ANGLES, d):
                ex, ey = x + math.cos(psi + ang) * dist, y + math.sin(psi + ang) * dist
                c.line([px(x, y), px(ex, ey)], fill=(120, 220, 255, 90), width=1)
        bx, by = px(x, y)
        car = [(bx - 2.6 * ppu, by - 1.3 * ppu), (bx + 2.6 * ppu, by - 1.3 * ppu),
               (bx + 2.6 * ppu, by + 1.3 * ppu), (bx - 2.6 * ppu, by + 1.3 * ppu)]
        c.poly(rotate(car, -psi, bx, by), fill=(150, 80, 80) if crashed else (235, 70, 60),
               outline=(30, 30, 30), width=1)
        nose = [(bx + 0.9 * ppu, by - 1.0 * ppu), (bx + 2.2 * ppu, by - 1.0 * ppu),
                (bx + 2.2 * ppu, by + 1.0 * ppu), (bx + 0.9 * ppu, by + 1.0 * ppu)]
        c.poly(rotate(nose, -psi, bx, by), fill=(40, 44, 60))

        c.rect(0, 0, W, 26, fill=(20, 24, 34))
        c.text(8, 5, label or self.title, fill=(235, 238, 250), size=13)
        c.text(W - 8, 7, f"laps {self.metric[i]:4.2f}   {self.v[i] * 3.6:3.0f} km/h", fill=(170, 180, 205),
               size=10, anchor="ra")
        if crashed:
            c.text(W / 2, H / 2 - 12, "CRASHED", fill=(255, 130, 120), size=20, anchor="ma")
        elif self.done[i]:
            c.text(W / 2, H / 2 - 12, f"{self.metric[i]:.2f} LAPS", fill=(150, 240, 170), size=20, anchor="ma")
        return c.finish()
