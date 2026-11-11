"""Small drawing helper on top of Pillow.

Frames are drawn at twice the final size and then shrunk, which smooths the
edges of shapes and text.
"""
import math

from PIL import Image, ImageDraw, ImageFont

SS = 2  # supersampling factor
_fonts = {}


def font(size):
    key = int(size * SS)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.load_default(size=key)
        except TypeError:  # very old Pillow
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]


class Canvas:
    def __init__(self, w, h, bg):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.img, "RGBA")

    def _p(self, pts):
        return [(x * SS, y * SS) for x, y in pts]

    def poly(self, pts, fill=None, outline=None, width=1):
        self.d.polygon(self._p(pts), fill=fill, outline=outline, width=int(width * SS))

    def line(self, pts, fill, width=1):
        self.d.line(self._p(pts), fill=fill, width=max(1, int(width * SS)), joint="curve")

    def circle(self, x, y, r, fill=None, outline=None, width=1):
        self.d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
                       fill=fill, outline=outline, width=max(1, int(width * SS)))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1):
        self.d.rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], fill=fill, outline=outline,
                         width=max(1, int(width * SS)))

    def text(self, x, y, s, fill=(255, 255, 255), size=12, anchor="la"):
        self.d.text((x * SS, y * SS), s, fill=fill, font=font(size), anchor=anchor)

    def finish(self):
        return self.img.resize((self.w, self.h), Image.LANCZOS)


def rotate(pts, ang, cx=0.0, cy=0.0):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]
