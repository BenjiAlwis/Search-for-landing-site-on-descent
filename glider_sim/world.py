"""Ground scene: procedural farmland terrain plus a marked runway.

Both are stored as image pyramids (BGR uint8) in metric ground coordinates so
the renderer can pick a mip level from each pixel's ground footprint.
"""
import cv2
import numpy as np


class Runway:
    def __init__(self, center_ne, heading_deg, length, width):
        self.center = np.asarray(center_ne, dtype=float)
        self.heading = np.radians(heading_deg)
        self.length = float(length)
        self.width = float(width)
        self.axis = np.array([np.cos(self.heading), np.sin(self.heading)])  # along (N, E)
        self.normal = np.array([-self.axis[1], self.axis[0]])                # right of axis

    def to_local(self, ne):
        d = np.asarray(ne, dtype=float) - self.center
        return d @ self.axis, d @ self.normal

    def corners_ne(self):
        l, w = self.length / 2, self.width / 2
        return np.array([self.center + sa * l * self.axis + sc * w * self.normal
                         for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def threshold(self, landing_dir):
        """Threshold point for landing in +axis (landing_dir=+1) or -axis (-1)."""
        return self.center - landing_dir * self.length / 2 * self.axis

    def designators(self):
        hdg = np.degrees(self.heading) % 360
        num = lambda h: f"{int(round(h / 10.0)) % 36 or 36:02d}"
        return num(hdg), num(hdg + 180)


def _pyramid(img, levels=10):
    pyr = [img]
    while len(pyr) < levels and min(pyr[-1].shape[:2]) > 16:
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _value_noise(rng, size, cells):
    small = rng.random((cells, cells)).astype(np.float32)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)


def make_terrain(rng, extent_m=10240.0, res_m=2.5):
    n = int(extent_m / res_m)
    img = np.empty((n, n, 3), np.float32)
    img[:] = (60, 120, 95)  # BGR base grass

    palette = np.array([(55, 125, 90), (40, 105, 70), (70, 140, 120), (80, 150, 165),
                        (60, 110, 140), (50, 95, 60), (95, 160, 175), (45, 130, 105)], np.float32)
    for _ in range(int(2.2 * (extent_m / 250) ** 2)):
        c = rng.uniform(0, n, 2)
        w, h = rng.uniform(60, 300, 2) / res_m
        box = cv2.boxPoints(((c[0], c[1]), (w, h), rng.uniform(0, 180))).astype(np.int32)
        cv2.fillConvexPoly(img, box, palette[rng.integers(len(palette))].tolist())

    # Tree lines / woods
    for _ in range(int(0.6 * (extent_m / 100) ** 2 / 10)):
        c = rng.uniform(0, n, 2).astype(int)
        cv2.circle(img, (int(c[0]), int(c[1])), int(rng.uniform(4, 25) / res_m), (35, 70, 40), -1)

    # Roads (distractors for the detector: long, grey, narrower than a runway)
    for _ in range(10):
        p0, p1 = rng.uniform(0, n, 2), rng.uniform(0, n, 2)
        cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), (95, 95, 98),
                 max(1, int(round(6 / res_m))), cv2.LINE_AA)

    shade = 0.75 + 0.35 * _value_noise(rng, n, 64) + 0.15 * (rng.random((n, n), np.float32) - 0.5)
    img *= shade[..., None]
    return np.clip(img, 0, 255).astype(np.uint8), res_m, extent_m


def make_runway_texture(rwy: Runway, rng, res_m=0.1):
    L, W = rwy.length, rwy.width
    nx, ny = int(round(L / res_m)), int(round(W / res_m))
    m = lambda v: int(round(v / res_m))
    img = np.empty((ny, nx, 3), np.float32)
    img[:] = (72, 72, 75)
    img *= (0.85 + 0.25 * rng.random((ny, nx), np.float32))[..., None]
    white = (235, 235, 235)

    edge = m(0.9)
    img[:edge] = white
    img[-edge:] = white

    # Threshold "piano keys", designators and aiming-point markers at both ends
    des = rwy.designators()
    for end in (0, 1):
        sign = 1 if end == 0 else -1
        x0 = m(6) if end == 0 else nx - m(6)
        n_keys = max(4, int(W // 2.6) & ~1)
        key_w = (W - 3.0) / (2 * n_keys - 1)
        for k in range(n_keys):
            y = 1.5 + 2 * k * key_w
            xa, xb = sorted((x0, x0 + sign * m(30)))
            img[m(y):m(y + key_w), xa:xb] = white

        # Runway number, read by a pilot approaching this end
        txt = des[end]
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 10)
        canvas = np.zeros((th + 20, tw + 10), np.uint8)
        cv2.putText(canvas, txt, (5, th + 10), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 10)
        canvas = cv2.resize(canvas, (m(W * 0.6), m(18)))  # tall characters, 18 m long
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE if end == 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
        xs = x0 + sign * m(40)
        xa = xs if end == 0 else xs - canvas.shape[1]
        ya = (ny - canvas.shape[0]) // 2
        roi = img[ya:ya + canvas.shape[0], xa:xa + canvas.shape[1]]
        roi[canvas > 128] = white

        for side in (-1, 1):
            xa, xb = sorted((x0 + sign * m(70), x0 + sign * m(85)))
            yc = ny // 2 + side * m(W * 0.25)
            img[yc - m(1.5):yc + m(1.5), xa:xb] = white

    # Centreline dashes between the numbers
    x = m(6 + 30 + 40 + 18 + 6)
    while x + m(20) < nx - m(6 + 30 + 40 + 18 + 6):
        img[ny // 2 - m(0.45):ny // 2 + m(0.45), x:x + m(20)] = white
        x += m(32)

    # Tyre rubber in touchdown zones
    for end in (0, 1):
        xa = m(60) if end == 0 else nx - m(160)
        img[:, xa:xa + m(100)] *= (0.85 + 0.15 * _value_noise(rng, max(nx, ny), 40)[:ny, :m(100)])[..., None]

    return np.clip(img, 0, 255).astype(np.uint8), res_m


class World:
    def __init__(self, runway: Runway, seed=0, terrain_extent_m=10240.0, terrain_res_m=2.5):
        rng = np.random.default_rng(seed)
        self.runway = runway
        terrain, self.terrain_res, self.terrain_extent = make_terrain(rng, terrain_extent_m, terrain_res_m)
        rtex, self.runway_res = make_runway_texture(runway, rng)
        self.terrain_pyr = _pyramid(terrain)
        self.runway_pyr = _pyramid(rtex)
        # Tiling high-frequency detail (grass clumps, soil) so the ground stays sharp up close
        n = 1024
        detail = (0.5 * _value_noise(rng, n, 256) + 0.3 * _value_noise(rng, n, 64)
                  + 0.2 * rng.random((n, n), np.float32))
        detail = (detail - detail.mean()) / (detail.std() + 1e-6)
        self.detail_res = 0.08  # m per detail texel (82 m tile)
        self.detail_pyr = _pyramid(np.repeat(np.clip(128 + 40 * detail, 0, 255).astype(np.uint8)[..., None], 3, 2))

    @staticmethod
    def _remap(img, mx, my, border):
        n = mx.size
        cols = 1024
        rows = -(-n // cols)
        pad = rows * cols - n
        mx = np.pad(mx, (0, pad)).reshape(rows, cols).astype(np.float32)
        my = np.pad(my, (0, pad)).reshape(rows, cols).astype(np.float32)
        out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=border)
        return out.reshape(-1, 3)[:n]

    def _sample_pyr(self, pyr, res0, px, py, footprint, border):
        level = np.clip(np.floor(np.log2(np.maximum(footprint / res0, 1.0))), 0, len(pyr) - 1).astype(int)
        out = np.empty((px.size, 3), np.uint8)
        for lv in np.unique(level):
            sel = level == lv
            s = 0.5 ** lv
            out[sel] = self._remap(pyr[lv], (px[sel] + 0.5) * s - 0.5, (py[sel] + 0.5) * s - 0.5, border)
        return out

    def sample(self, ne, footprint):
        """Ground colour (BGR uint8) at (N,2) north/east points, footprint in metres."""
        n, e = ne[:, 0], ne[:, 1]
        half = self.terrain_extent / 2
        tx = (e + half) / self.terrain_res
        ty = (half - n) / self.terrain_res
        out = self._sample_pyr(self.terrain_pyr, self.terrain_res, tx, ty, footprint, cv2.BORDER_REFLECT_101)

        near = footprint < 1.5  # blend detail in where texels are visibly larger than pixels
        if near.any():
            dx = e[near] / self.detail_res
            dy = -n[near] / self.detail_res
            d = self._sample_pyr(self.detail_pyr, self.detail_res, dx, dy, footprint[near], cv2.BORDER_WRAP)
            fade = np.clip((1.5 - footprint[near]) / 1.0, 0, 1)[:, None]
            gain = 1 + fade * (d[:, :1].astype(np.float32) - 128) / 128 * 0.6
            out[near] = np.clip(out[near] * gain, 0, 255).astype(np.uint8)

        along, cross = self.runway.to_local(ne)
        on = (np.abs(along) <= self.runway.length / 2) & (np.abs(cross) <= self.runway.width / 2)
        if on.any():
            rx = (along[on] + self.runway.length / 2) / self.runway_res
            ry = (cross[on] + self.runway.width / 2) / self.runway_res
            out[on] = self._sample_pyr(self.runway_pyr, self.runway_res, rx, ry, footprint[on],
                                       cv2.BORDER_REPLICATE)
        return out
