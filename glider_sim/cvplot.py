"""Minimal line-plot drawing with OpenCV (fast enough to redraw every frame)."""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (28, 24, 22)
AXIS = (150, 150, 150)
GRID = (55, 52, 50)
TEXT = (225, 225, 225)


def nice_ticks(lo, hi, n=5):
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = mag * min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw))
    start = np.ceil(lo / step) * step
    return np.arange(start, hi + step * 0.5, step)


def fmt(v):
    if abs(v) >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 1 else f"{v:.1f}"


class CvPlot:
    """Axes drawn into `img` inside rect (x, y, w, h)."""

    def __init__(self, img, rect, xlim, ylim, title="", xlabel="", ylabel="", scale=0.38):
        self.img = img
        x, y, w, h = rect
        self.s = scale
        self.ox, self.oy = x + 46, y + (22 if title else 8)
        self.w, self.h = w - 46 - 10, h - (22 if title else 8) - (34 if xlabel else 20)
        self.xlim, self.ylim = xlim, ylim
        cv2.rectangle(img, (self.ox, self.oy), (self.ox + self.w, self.oy + self.h), (36, 32, 30), -1)
        for tx in nice_ticks(*xlim):
            px = self.X(tx)
            cv2.line(img, (px, self.oy), (px, self.oy + self.h), GRID, 1)
            self.text(fmt(tx), (px - 8, self.oy + self.h + 14), AXIS)
        for ty in nice_ticks(*ylim):
            py = self.Y(ty)
            cv2.line(img, (self.ox, py), (self.ox + self.w, py), GRID, 1)
            self.text(fmt(ty), (x + 4, py + 4), AXIS)
        cv2.rectangle(img, (self.ox, self.oy), (self.ox + self.w, self.oy + self.h), AXIS, 1)
        if title:
            self.text(title, (x + 6, y + 15), TEXT, 1.15)
        if xlabel:
            self.text(xlabel, (self.ox + self.w // 2 - 4 * len(xlabel), self.oy + self.h + 29), AXIS)
        if ylabel:
            self.text(ylabel, (self.ox + 4, self.oy + 12), AXIS)

    def X(self, x):
        x0, x1 = self.xlim
        return int(round(self.ox + (np.clip(x, x0, x1) - x0) / (x1 - x0) * self.w))

    def Y(self, y):
        y0, y1 = self.ylim
        return int(round(self.oy + self.h - (np.clip(y, y0, y1) - y0) / (y1 - y0) * self.h))

    def _pts(self, xs, ys):
        return np.array([[self.X(a), self.Y(b)] for a, b in zip(xs, ys)], np.int32)

    def line(self, xs, ys, color, thickness=1, dashed=False):
        if len(xs) < 2:
            return
        pts = self._pts(xs, ys)
        if not dashed:
            cv2.polylines(self.img, [pts], False, color, thickness, cv2.LINE_AA)
            return
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(int(np.linalg.norm(b - a) / 6), 1)
            for i in range(0, n, 2):
                p = a + (b - a) * i / n
                q = a + (b - a) * min(i + 1, n) / n
                cv2.line(self.img, tuple(p.astype(int)), tuple(q.astype(int)), color, thickness, cv2.LINE_AA)

    def fill_between(self, xs, y_lo, y_hi, color, alpha=0.25):
        poly = np.vstack([self._pts(xs, y_lo), self._pts(xs[::-1], y_hi[::-1])])
        overlay = self.img.copy()
        cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0, self.img)

    def point(self, x, y, color, r=4):
        cv2.circle(self.img, (self.X(x), self.Y(y)), r, color, -1, cv2.LINE_AA)

    def text(self, s, org, color=TEXT, rel=1.0):
        cv2.putText(self.img, s, (int(org[0]), int(org[1])), FONT, self.s * rel, color, 1, cv2.LINE_AA)

    def legend(self, items, corner="tr"):
        x = self.ox + self.w - 8 - max(len(s) for s, _ in items) * 7 - 18 if corner == "tr" else self.ox + 8
        y = self.oy + 14
        for s, col in items:
            cv2.line(self.img, (x, y - 4), (x + 14, y - 4), col, 2, cv2.LINE_AA)
            self.text(s, (x + 18, y), TEXT)
            y += 14
