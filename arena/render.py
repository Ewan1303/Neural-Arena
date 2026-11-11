"""Turning trained agents into pictures: episode GIFs and NEAT network diagrams."""
from PIL import Image

from .envs import make_env


def record_episode(env_name, policy, seed, label=None, frame_size=None, hold=12, every=1):
    """Run one episode and return the frames (Pillow images). The last frame is
    repeated `hold` times so the outcome stays on screen."""
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


def save_gif(frames, path, ms=50, colors=64):
    """Write an animated GIF. All frames share one colour palette, which keeps the file small."""
    sample = frames[len(frames) // 2].quantize(colors=colors, method=Image.Quantize.MEDIANCUT,
                                               dither=Image.Dither.NONE)
    pal = [f.quantize(palette=sample, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=False,
                disposal=1)
