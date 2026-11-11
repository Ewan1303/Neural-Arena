"""Turning trained agents into pictures: recording episodes and writing GIFs."""
import numpy as np
from PIL import Image

from .envs import make_env


def record_episode(env_name, policy, seed, label=None, frame_size=None, hold=12, every=1):
    """Run one episode and return (frames, return, metric). The last frame is
    repeated so the outcome stays on screen. `every` keeps one frame in that many."""
    env = make_env(env_name, 1)
    if frame_size:
        env.frame_size = frame_size
    obs = env.reset(seed)
    policy.reset(1)
    frames = [env.render_frame(0, label)]
    step = 0
    while not env.done.all():
        obs, _, _, _ = env.step(policy.act(obs))
        step += 1
        if step % every == 0 or env.done.all():
            frames.append(env.render_frame(0, label))
    frames += [frames[-1]] * max(1, hold // every)
    return frames, float(env.ep_return[0]), float(env.metric[0])


def episode_metric(env_name, policy, seed):
    """Score of one episode without drawing it."""
    env = make_env(env_name, 1)
    obs = env.reset(seed)
    policy.reset(1)
    while not env.done.all():
        obs, _, _, _ = env.step(policy.act(obs))
    return float(env.metric[0])


def side_by_side(columns, gap=4, bg=(12, 14, 20)):
    """Join several frame lists into one, left to right. Shorter ones hold their last frame."""
    n = max(len(c) for c in columns)
    w = sum(c[0].width for c in columns) + gap * (len(columns) - 1)
    h = max(c[0].height for c in columns)
    out = []
    for k in range(n):
        frame = Image.new("RGB", (w, h), bg)
        x = 0
        for c in columns:
            frame.paste(c[min(k, len(c) - 1)], (x, 0))
            x += c[0].width + gap
        out.append(frame)
    return out


def save_gif(frames, path, ms=50, colors=160):
    """Write an animated GIF that stays small.

    All frames share one colour palette, and every frame after the first only
    stores the pixels that changed: unchanged pixels are written as "transparent",
    which lets the previous frame show through and compresses to almost nothing.
    """
    colors = min(colors, 255)                      # index 255 is kept for "unchanged"
    # Build the palette from frames spread over the whole clip (including the last one, where
    # the result text appears), and with a method that keeps small patches of strong colour
    # such as the red car, rather than spending every entry on the background.
    picks = [frames[int(round(i * (len(frames) - 1) / 7))] for i in range(8)]
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
    for i, f in enumerate(picks):
        sheet.paste(f, (0, i * f.height))
    sample = sheet.quantize(colors=colors, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE)
    palette = sample.getpalette()
    palette = palette + [0] * (768 - len(palette))
    index = [np.asarray(f.quantize(palette=sample, dither=Image.Dither.NONE), dtype=np.uint8) for f in frames]
    out = []
    for k, a in enumerate(index):
        if k:
            a = np.where(a == index[k - 1], 255, a).astype(np.uint8)
        im = Image.fromarray(a, mode="P")
        im.putpalette(palette)
        out.append(im)
    out[0].save(path, save_all=True, append_images=out[1:], duration=ms, loop=0, optimize=False,
                disposal=1, transparency=255)
