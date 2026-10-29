"""Multi-panel dashboard for exploring the environment.

A 4x3 grid: one large main tile (2x2) plus eight small tiles. Any panel can be
moved to the main tile (number keys 1-9 or a mouse click on its tile).

  1 outside       external camera (chase / tower / side / overhead) with the glider
  2 eye           onboard fisheye, true runway (green) vs detected (red)
  3 birds-eye     fisheye re-projected onto the ground plane (detector step 1)
  4 saturation    HSV saturation channel, threshold contour (detector step 2)
  5 segmentation  asphalt / paint classes and the cleaned mask (steps 2-3)
  6 candidates    fitted rectangles with accept / reject reasons (step 4)
  7 map           top-down map, flight path, energy orbit, FAF, aim point
  8 glide         altitude vs distance-to-go against the glide cone + equations
  9 timeline      glide angles, flap, detections and runway-estimate error vs time
"""
from collections import deque

import cv2
import numpy as np

from .camera import OmniCamera, Renderer
from .cvplot import BG, CvPlot
from .rotations import rot_from_quat

TILE_W, TILE_H = 400, 300
MAIN_W, MAIN_H = 2 * TILE_W, 2 * TILE_H
STATUS_H = 30  # status bar under the grid
CANVAS_W, CANVAS_H = 4 * TILE_W, 3 * TILE_H + STATUS_H
MAP_SIZE = 600
DBG_SIZE = 600  # detector images are shrunk to this once per frame for display
PLOT_TOP = 30   # keep plots clear of the tile label
VIEWS = ("chase", "tower", "side", "overhead")
PANELS = ("outside", "eye", "birds-eye", "saturation", "segmentation", "candidates", "map", "glide", "timeline")
SLOTS = [(800, 0), (1200, 0), (800, 300), (1200, 300), (0, 600), (400, 600), (800, 600), (1200, 600)]
PHASE_COLORS = {"SEARCH": (160, 160, 160), "ORBIT": (230, 160, 60), "FINAL": (60, 160, 240),
                "FLARE": (80, 80, 240), "FREE": (200, 200, 200)}
GREEN, RED, YELLOW, CYAN, MAGENTA = (80, 220, 80), (60, 60, 240), (60, 220, 240), (240, 220, 60), (230, 80, 230)


def put_label(img, text, org, scale=0.6):
    """White text with a black outline (offset copies keep glyph spacing identical)."""
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0), (-2, 0), (0, -2)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def look_at(cam_pos, target):
    """R_ned_cam for a camera at cam_pos looking at target, image 'down' ~ world down."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, 1.0])
    x = np.cross(down, z)
    if np.linalg.norm(x) < 1e-6:  # looking straight down: keep north at image top
        x = np.array([0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def glider_model(params):
    """Coloured body-frame polygons (FRD) of a simple glider sized from the spec."""
    b, c = params.b, params.c
    L = max(0.55 * b, 4 * c)
    wing = np.array([[0.25 * c, -b / 2, 0], [0.25 * c, b / 2, 0], [-0.75 * c, b / 2, 0], [-0.75 * c, -b / 2, 0]])
    ht = np.array([[-0.62 * L, -0.17 * b, 0], [-0.62 * L, 0.17 * b, 0], [-0.62 * L - 0.7 * c, 0.17 * b, 0],
                   [-0.62 * L - 0.7 * c, -0.17 * b, 0]])
    vt = np.array([[-0.6 * L, 0, 0], [-0.62 * L - 0.7 * c, 0, -0.2 * b], [-0.62 * L - 0.7 * c, 0, 0]])
    w = 0.05 * b
    fus = np.array([[0.4 * L, 0, 0], [0.1 * L, w, 0], [-0.62 * L - 0.7 * c, 0.3 * w, 0],
                    [-0.62 * L - 0.7 * c, -0.3 * w, 0], [0.1 * L, -w, 0]])
    return [(wing, (240, 240, 245)), (ht, (225, 225, 230)), (fus, (60, 60, 200)), (vt, (40, 40, 180))]


def fit(img, w, h, interp=cv2.INTER_AREA):
    """Letterbox img into (h, w); returns canvas and (scale, x0, y0) for mapping coords."""
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    out = np.full((h, w, 3), BG, np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    out[y0:y0 + nh, x0:x0 + nw] = small if small.ndim == 3 else small[..., None]
    return out, (s, x0, y0)


def placeholder(w, h, lines):
    img = np.full((h, w, 3), BG, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (20, h // 2 - 10 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def runway_outline(center, heading, length, width, n=25):
    """Ground points (N, 3) along the edges of a runway rectangle."""
    ax = np.array([np.cos(heading), np.sin(heading)])
    nm = np.array([-ax[1], ax[0]])
    c = [center + sa * length / 2 * ax + sc * width / 2 * nm for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    s = np.linspace(0, 1, n, endpoint=False)[:, None]
    pts = np.vstack([c[i] + s * (c[(i + 1) % 4] - c[i]) for i in range(4)])
    return np.column_stack([pts, np.zeros(len(pts))])


class Explorer:
    def __init__(self, world, params, fisheye_cam: OmniCamera, runway, seed=0, render_scale=1.0, airspeed=12.0):
        self.world = world
        self.runway = runway
        self.fisheye = fisheye_cam
        self.fish_renderer = Renderer(fisheye_cam, world, seed)
        base = {"projection": "pinhole", "mount_rpy_deg": [0, 0, 0], "visibility_m": 9000}
        self.ext = {}  # (lens, size) -> (camera at display size, renderer at render size)
        for name, fov in (("wide", 70), ("tele", 22)):  # tele = tower spotter lens
            for size, (w, h) in (("main", (MAIN_W, MAIN_H)), ("tile", (TILE_W, TILE_H))):
                cam = OmniCamera({**base, "fov_deg": fov, "width_px": w, "height_px": h})
                lo = OmniCamera({**base, "fov_deg": fov, "width_px": int(w * render_scale),
                                 "height_px": int(h * render_scale)}) if render_scale != 1.0 else cam
                self.ext[name, size] = (cam, Renderer(lo, world, seed))
        self._dbg_src = None
        self._dbg_small = None
        self.model = glider_model(params)
        f, tan_g = params.glide_slope_table(airspeed)
        self.tan_min, self.tan_max = float(tan_g[0]), float(tan_g[-1])
        self.view = 0
        self.zoom = 1.0
        self.main = 0
        self.trail = []
        self.hist = deque(maxlen=3000)
        self._build_map()

    # ================================================================== state / input
    def record(self, st):
        info = st.get("info") or {}
        nav = st["nav"]
        S, tan_req = self.distance_to_go(st)
        est = st.get("estimate")
        err = np.linalg.norm(est.center - self.runway.center) if est is not None else np.nan
        vel = nav.vel
        gamma = np.degrees(np.arctan2(vel[2], max(np.hypot(vel[0], vel[1]), 0.1)))
        if self.hist:  # light smoothing: nav velocity is noisy
            gamma = 0.7 * self.hist[-1]["gamma"] + 0.3 * gamma
        self.hist.append({
            "t": nav.t, "h": nav.altitude, "S": S, "tan_req": tan_req, "flap": st.get("flap", 0.0),
            "gamma": gamma,
            "det": st.get("detection") is not None, "err": err, "phase": info.get("phase", "FREE"),
        })
        if not self.trail or np.linalg.norm(self.trail[-1] - st["pos"][:2]) > 2.0:
            self.trail.append(st["pos"][:2].copy())

    def distance_to_go(self, st):
        info = st.get("info") or {}
        h = st["nav"].altitude
        if "S" in info:
            S = info["S"]
        else:  # free mode / flare: straight line to the aim point
            aim = info.get("aim", self.runway.threshold(1) + 40 * self.runway.axis)
            S = float(np.linalg.norm(st["nav"].pos[:2] - aim))
        return S, h / max(S, 1.0)

    def handle_key(self, key):
        if key == ord("v"):
            self.view = (self.view + 1) % len(VIEWS)
        elif key in (ord("+"), ord("=")):
            self.zoom = max(0.3, self.zoom / 1.25)
        elif key in (ord("-"), ord("_")):
            self.zoom = min(8.0, self.zoom * 1.25)
        elif ord("1") <= key <= ord("9"):
            self.main = key - ord("1")
        elif key == 9:  # Tab
            self.main = (self.main + 1) % len(PANELS)

    def handle_click(self, x, y):
        others = [i for i in range(len(PANELS)) if i != self.main]
        for slot, idx in zip(SLOTS, others):
            if slot[0] <= x < slot[0] + TILE_W and slot[1] <= y < slot[1] + TILE_H:
                self.main = idx
                return

    # ================================================================== compose
    def compose(self, st):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        others = [i for i in range(len(PANELS)) if i != self.main]
        canvas[:MAIN_H, :MAIN_W] = self.panel(self.main, st, MAIN_W, MAIN_H)
        for (x, y), idx in zip(SLOTS, others):
            canvas[y:y + TILE_H, x:x + TILE_W] = self.panel(idx, st, TILE_W, TILE_H)
            put_label(canvas, f"{idx + 1} {PANELS[idx]}", (x + 8, y + 20), 0.5)
            cv2.rectangle(canvas, (x, y), (x + TILE_W - 1, y + TILE_H - 1), (0, 0, 0), 2)
        title = f"{self.main + 1} {PANELS[self.main]}"
        if PANELS[self.main] == "outside":
            title += f" - {VIEWS[self.view]}"
        put_label(canvas, title, (12, 24), 0.6)
        cv2.rectangle(canvas, (0, 0), (MAIN_W - 1, MAIN_H - 1), (0, 0, 0), 2)

        y = 3 * TILE_H
        canvas[y:] = (12, 12, 12)
        status = "   |   ".join(st.get("hud", []))
        cv2.putText(canvas, status, (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        hint = "click tile / 1-9: enlarge   v: camera   +/-: zoom   q: quit"
        cv2.putText(canvas, hint, (CANVAS_W - 8 - 8 * len(hint), y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)
        return canvas

    def panel(self, idx, st, w, h):
        fn = getattr(self, "p_" + PANELS[idx].replace("-", "_"))
        out = fn(st, w, h)
        return out if out.shape[:2] == (h, w) else cv2.resize(out, (w, h))

    # ================================================================== panels
    def p_outside(self, st, w, h):
        pos, R_nb = st["pos"], st["R"]
        cam_pos, R_cam = self.external_pose(pos, R_nb, st.get("info"))
        lens = "tele" if VIEWS[self.view] == "tower" else "wide"
        cam, ren = self.ext[lens, "main" if w > TILE_W else "tile"]
        img = ren.render(cam_pos, R_cam)
        if img.shape[1] != cam.W:
            img = cv2.resize(img, (cam.W, cam.H), interpolation=cv2.INTER_LINEAR)
        self.draw_glider(img, cam, cam_pos, R_cam, pos, R_nb)
        return img

    def p_eye(self, st, w, h):
        frame = st.get("frame")
        if frame is None:
            frame = self.fish_renderer.render(*self.fisheye.pose(st["pos"], st["R"]))
        out, (s, x0, y0) = fit(frame, w, h)
        th = max(1, int(round(2 * s)))
        truth_pose = self.fisheye.pose(st["pos"], st["R"])
        rw = self.runway
        uv, ok = self.fisheye.world_to_pixel(runway_outline(rw.center, rw.heading, rw.length, rw.width), *truth_pose)
        if ok.any():
            self._curve(out, uv, ok, s, x0, y0, GREEN, th)
        det = st.get("detection")
        if det is not None:
            nav = st["nav"]
            uv, ok = self.fisheye.world_to_pixel(runway_outline(det.center_ne, det.axis_heading, det.length,
                                                                det.width), *self.fisheye.pose(nav.pos, nav.R))
            if ok.any():
                self._curve(out, uv, ok, s, x0, y0, RED, th)
        self._legend(out, [("truth", GREEN), ("detected", RED)])
        return out

    @staticmethod
    def _curve(img, uv, ok, s, x0, y0, color, th):
        pts = (uv * s + [x0, y0]).astype(np.int32)
        # draw only segments whose both ends are visible (outline may leave the fisheye circle)
        for a, b, oa, ob in zip(pts, np.roll(pts, -1, 0), ok, np.roll(ok, -1)):
            if oa and ob:
                cv2.line(img, tuple(a), tuple(b), color, th, cv2.LINE_AA)

    def _dbg(self, st):
        """Detector debug data shrunk to DBG_SIZE once per frame (pixel coords rescaled)."""
        d = st.get("debug")
        if d is None:
            return None
        if d is not self._dbg_src:
            f = DBG_SIZE / d["bev"].shape[0]
            size = (DBG_SIZE, DBG_SIZE)
            area = lambda a: cv2.resize(a, size, interpolation=cv2.INTER_AREA)
            near = lambda a: cv2.resize(a.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            self._dbg_small = {
                "bev": area(d["bev"]), "sat": area(d["sat"]), "valid": near(d["valid"]) > 0,
                "asphalt": area(d["asphalt"].astype(np.uint8) * 255) > 64,
                "paint": area(d["paint"].astype(np.uint8) * 255) > 64,
                "mask": near(d["mask"]), "half": d["half"], "res": d["res"] / f, "kernels": d["kernels"],
                "box_px": None if d["box_px"] is None else d["box_px"] * f,
                "candidates": [(box * f, verdict) for box, verdict in d["candidates"]],
            }
            self._dbg_src = d
        return self._dbg_small

    def _bev_overlay(self, out, dbg, s, x0, y0):
        """Aircraft marker and scale bar on a bird's-eye panel."""
        n = dbg["bev"].shape[0]
        c = (int(x0 + n / 2 * s), int(y0 + n / 2 * s))
        cv2.drawMarker(out, c, (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        bar_m = 100 if dbg["half"] < 800 else 500
        bar_px = int(bar_m / dbg["res"] * s)
        cv2.line(out, (x0 + 10, y0 + int(n * s) - 12), (x0 + 10 + bar_px, y0 + int(n * s) - 12), (255, 255, 255), 2)
        put_label(out, f"{bar_m} m", (x0 + 10, y0 + int(n * s) - 18), 0.4)

    def p_birds_eye(self, st, w, h):
        dbg = self._dbg(st)
        if dbg is None:
            return placeholder(w, h, ["no detector output", "(oracle mode: press c to run the CV)"])
        out, (s, x0, y0) = fit(dbg["bev"], w, h)
        if dbg["box_px"] is not None:
            cv2.polylines(out, [(dbg["box_px"] * s + [x0, y0]).astype(np.int32)], True, RED, 2, cv2.LINE_AA)
        self._bev_overlay(out, dbg, s, x0, y0)
        put_label(out, f"ground plane +/-{dbg['half']:.0f} m, {dbg['res']:.2f} m/px, north up",
                  (x0 + 8, y0 + 40 if w > TILE_W else y0 + 44), 0.42)
        return out

    def p_saturation(self, st, w, h):
        dbg = self._dbg(st)
        if dbg is None:
            return placeholder(w, h, ["no detector output", "(oracle mode: press c to run the CV)"])
        sat = np.clip(dbg["sat"].astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(sat, cv2.COLORMAP_INFERNO)
        heat[~dbg["valid"]] = BG
        out, (s, x0, y0) = fit(heat, w, h)
        low = cv2.resize((dbg["sat"] < 40).astype(np.uint8) * 255, (int(heat.shape[1] * s), int(heat.shape[0] * s)),
                         interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(low, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, CYAN, 1, cv2.LINE_AA, offset=(x0, y0))
        self._legend(out, [("dark = grey (low S)", (40, 20, 10)), ("bright = colourful", (80, 200, 250)),
                           ("S < 40 boundary", CYAN)])
        return out

    def p_segmentation(self, st, w, h):
        dbg = self._dbg(st)
        if dbg is None:
            return placeholder(w, h, ["no detector output", "(oracle mode: press c to run the CV)"])
        out, (s, x0, y0) = fit(dbg["bev"], w, h)
        size = (int(dbg["bev"].shape[1] * s), int(dbg["bev"].shape[0] * s))
        roi = out[y0:y0 + size[1], x0:x0 + size[0]]
        roi[:] = (roi * 0.35).astype(np.uint8)
        for key, col in (("asphalt", (220, 130, 40)), ("paint", (80, 240, 250))):
            m = cv2.resize(dbg[key].astype(np.uint8), size, interpolation=cv2.INTER_AREA) > 0
            roi[m] = col
        clean = cv2.resize(dbg["mask"], size, interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cnts, -1, MAGENTA, 1, cv2.LINE_AA)
        k, ko = dbg["kernels"]
        self._legend(out, [("asphalt: S<40, 35<V<140", (220, 130, 40)), ("paint: S<45, V>185", (80, 240, 250)),
                           (f"mask after close {k}px / open {ko}px", MAGENTA)])
        return out

    def p_candidates(self, st, w, h):
        dbg = self._dbg(st)
        if dbg is None:
            return placeholder(w, h, ["no detector output", "(oracle mode: press c to run the CV)"])
        gray = cv2.cvtColor((dbg["mask"] * 0.45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out, (s, x0, y0) = fit(gray, w, h)
        chosen = dbg["box_px"]
        cands = sorted(dbg["candidates"], key=lambda c: -cv2.contourArea(c[0]))[:14]
        n_rej = 0
        for box, verdict in cands:
            is_best = chosen is not None and np.allclose(box, chosen)
            col = GREEN if is_best else (YELLOW if verdict.startswith("ok") else RED)
            n_rej += col == RED
            pts = (box * s + [x0, y0]).astype(np.int32)
            cv2.polylines(out, [pts], True, col, 2 if is_best else 1, cv2.LINE_AA)
            c = pts.mean(0).astype(int)
            cv2.putText(out, ("CHOSEN " if is_best else "") + verdict, (int(c[0]) + 6, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36 if w <= TILE_W else 0.45, col, 1, cv2.LINE_AA)
        self._legend(out, [(f"{len(dbg['candidates'])} shapes, {n_rej} rejected", (200, 200, 200)),
                           ("chosen runway", GREEN), ("plausible", YELLOW), ("rejected (reason)", RED)])
        return out

    def p_map(self, st, w, h):
        m = self.map_base.copy()
        if len(self.trail) > 1:
            cv2.polylines(m, [self._to_map(np.array(self.trail)).astype(np.int32)], False, (0, 220, 255), 2,
                          cv2.LINE_AA)
        info = st.get("info") or {}
        if "orbit_center" in info:
            oc = self._to_map(info["orbit_center"])[0]
            r = info.get("orbit_radius", 60.0) * MAP_SIZE / (2 * self.map_half)
            cv2.circle(m, tuple(oc.astype(int)), max(int(r), 2), (255, 180, 60), 1, cv2.LINE_AA)
            for key, col, lab in (("faf", (255, 180, 60), "FAF"), ("aim", (60, 255, 60), "aim")):
                q = self._to_map(info[key])[0].astype(int)
                cv2.drawMarker(m, tuple(q), col, cv2.MARKER_CROSS, 12, 2)
                put_label(m, lab, (int(q[0]) + 6, int(q[1]) - 6), 0.4)
        est = st.get("estimate")
        if est is not None:
            q = self._to_map(est.center)[0]
            cv2.circle(m, tuple(q.astype(int)), 6, RED, 1, cv2.LINE_AA)
        p = self._to_map(st["pos"][:2])[0]
        yaw = np.arctan2(st["R"][1, 0], st["R"][0, 0])
        fwd = np.array([np.sin(yaw), -np.cos(yaw)])
        side = np.array([-fwd[1], fwd[0]])
        tri = np.array([p + 12 * fwd, p - 7 * fwd + 6 * side, p - 7 * fwd - 6 * side])
        cv2.fillConvexPoly(m, tri.astype(np.int32), (255, 255, 255), cv2.LINE_AA)
        out, _ = fit(m, w, h)
        self._legend(out, [("flight path", (0, 220, 255)), ("energy orbit / FAF", (255, 180, 60)),
                           ("aim point", (60, 255, 60)), ("runway estimate", RED)])
        return out

    def p_glide(self, st, w, h):
        img = np.full((h, w, 3), BG, np.uint8)
        S, tan_req = self.distance_to_go(st)
        hcur = st["nav"].altitude
        recent = [r for r in self.hist if r["t"] > st["nav"].t - 40] or [{"S": S, "h": hcur}]
        xmax = max(max(r["S"] for r in recent), S, 100) * 1.15
        ymax = max(max(r["h"] for r in recent), hcur, 10) * 1.2
        big = w > TILE_W
        eq_h = 92 if big else 0
        p = CvPlot(img, (0, PLOT_TOP, w, h - eq_h - PLOT_TOP), (0, xmax), (0, ymax),
                   "altitude vs distance-to-go (glide cone)",
                   "distance to go S [m]", "h [m]", 0.42 if big else 0.34)
        xs = np.array([0, xmax])
        p.fill_between(xs, xs * self.tan_min, xs * self.tan_max, (80, 160, 80), 0.18)
        p.line(xs, xs * self.tan_min, GREEN, 1)
        p.line(xs, xs * self.tan_max, RED, 1)
        p.line(xs, xs * 0.8 * self.tan_max, (60, 160, 240), 1, dashed=True)
        p.line(xs, xs * tan_req, (255, 255, 255), 1, dashed=True)
        pts = [(r["S"], r["h"], r["phase"]) for r in recent]
        for (s0, h0, ph), (s1, h1, _) in zip(pts[:-1], pts[1:]):
            if abs(s1 - s0) < 0.2 * xmax:  # skip jumps when the orbit count changes
                p.line([s0, s1], [h0, h1], PHASE_COLORS.get(ph, (200, 200, 200)), 2)
        p.point(S, hcur, (255, 255, 255), 5)
        p.legend([("flaps up (best glide)", GREEN), ("full flap (steepest)", RED),
                  ("0.8 x steepest: orbit limit", (60, 160, 240)), ("required slope now", (255, 255, 255))],
                 corner="tl")
        if big:
            info = st.get("info") or {}
            g = lambda t: np.degrees(np.arctan(t))
            n = info.get("orbits_left")
            lines = [
                f"tan(gamma_req) = h / S = {hcur:.1f} / {S:.0f} = {tan_req:.3f}    ->  gamma_req = {g(tan_req):.1f} deg",
                f"achievable: tan(gamma) in [{self.tan_min:.3f}, {self.tan_max:.3f}]  =  [{g(self.tan_min):.1f}, "
                f"{g(self.tan_max):.1f}] deg   (from the drag polar: tan(gamma) = CD/CL)",
                f"flap = interp(tan(gamma_req) -> polar table) = {100 * st.get('flap', 0):.0f} %",
                ("S = (orbit arc to FAF) + n * 2 pi R + final length;  n = smallest with h/S <= 0.8 tan(gamma_max)"
                 + (f"   ->  n = {n}" if n is not None else "")) if "S" in info else
                "S = straight-line distance to the aim point (no guidance plan in this mode)",
            ]
            for i, s in enumerate(lines):
                cv2.putText(img, s, (14, h - eq_h + 18 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1,
                            cv2.LINE_AA)
        return img

    def p_timeline(self, st, w, h):
        img = np.full((h, w, 3), BG, np.uint8)
        t_now = st["nav"].t
        window = 60.0
        rec = [r for r in self.hist if r["t"] > t_now - window]
        if len(rec) < 2:
            return placeholder(w, h, ["collecting data ..."])
        t = np.array([r["t"] for r in rec])
        xl = (max(t_now - window, t[0]), max(t_now, t[0] + 1))
        sc = 0.42 if w > TILE_W else 0.34
        top = CvPlot(img, (0, PLOT_TOP, w, h // 2 + 6 - PLOT_TOP), xl, (0, 40), "glide angle [deg] and flap [%/2.5]",
                     "", "", sc)
        g = lambda v: np.degrees(np.arctan(v))
        top.fill_between(np.array(xl), np.full(2, g(self.tan_min)), np.full(2, g(self.tan_max)), (80, 160, 80), 0.15)
        top.line(t, [g(r["tan_req"]) for r in rec], (255, 255, 255), 1)
        top.line(t, [r["gamma"] for r in rec], (60, 160, 240), 1)
        top.line(t, [100 * r["flap"] / 2.5 for r in rec], YELLOW, 1)
        top.legend([("required", (255, 255, 255)), ("actual", (60, 160, 240)), ("flap", YELLOW)])

        err = np.array([r["err"] for r in rec])
        emax = max(np.nanmax(err) if np.isfinite(err).any() else 10, 10) * 1.15
        bot = CvPlot(img, (0, h // 2 + 6, w, h - h // 2 - 6), xl, (0, emax), "runway estimate error [m]", "t [s]", "",
                     sc)
        ok = np.isfinite(err)
        if ok.sum() > 1:
            bot.line(t[ok], err[ok], MAGENTA, 1)
        for r in rec[::max(1, len(rec) // 150)]:
            x = bot.X(r["t"])
            cv2.line(img, (x, bot.oy + 2), (x, bot.oy + 7), GREEN if r["det"] else RED, 1)
        bot.legend([("est. vs true centre", MAGENTA), ("detected / missed", GREEN)])
        return img

    @staticmethod
    def _legend(img, items):
        h = img.shape[0]
        y = h - 12 - 16 * (len(items) - 1)
        for s, col in items:
            cv2.rectangle(img, (10, y - 9), (20, y + 1), col, -1)
            put_label(img, s, (26, y), 0.4)
            y += 16

    # ================================================================== map
    def _build_map(self, half_extent=1400.0):
        w = self.world
        self.map_half = half_extent
        self.map_center = self.runway.center.copy()
        lv = 1
        res = w.terrain_res * 2 ** lv
        img = w.terrain_pyr[lv]
        c = self.map_center
        col0 = (c[1] - half_extent + w.terrain_extent / 2) / res
        row0 = (w.terrain_extent / 2 - (c[0] + half_extent)) / res
        size = 2 * half_extent / res
        M = np.float32([[size / MAP_SIZE, 0, col0], [0, size / MAP_SIZE, row0]])
        base = cv2.warpAffine(img, M, (MAP_SIZE, MAP_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REFLECT_101)
        base = (base * 0.7).astype(np.uint8)
        rw = self._to_map(self.runway.corners_ne()).astype(np.int32)
        cv2.polylines(base, [rw], True, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.fillConvexPoly(base, rw, (40, 40, 40), cv2.LINE_AA)
        put_label(base, "RWY", tuple((rw.mean(0) + [10, -10]).astype(int)), 0.5)
        put_label(base, "N", (MAP_SIZE - 22, 24), 0.6)
        bar = int(500 * MAP_SIZE / (2 * half_extent))
        cv2.line(base, (MAP_SIZE - 20 - bar, 44), (MAP_SIZE - 20, 44), (255, 255, 255), 2)
        put_label(base, "500 m", (MAP_SIZE - 20 - bar, 38), 0.45)
        self.map_base = base

    def _to_map(self, ne):
        ne = np.atleast_2d(ne)
        s = MAP_SIZE / (2 * self.map_half)
        return np.column_stack([(ne[:, 1] - self.map_center[1] + self.map_half) * s,
                                (self.map_center[0] + self.map_half - ne[:, 0]) * s])

    # ================================================================== external view
    def external_pose(self, pos, R_nb, info=None):
        view = VIEWS[self.view]
        fwd = R_nb[:, 0].copy()
        fwd[2] = 0
        fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
        right = np.array([-fwd[1], fwd[0], 0.0])
        z = self.zoom
        if view == "chase":
            cam = pos - 9 * z * fwd + np.array([0, 0, -2.5 * z])
            target = pos + 10 * fwd
        elif view == "side":
            cam = pos - 12 * z * right + np.array([0, 0, -2.0 * z])
            target = pos
        elif view == "tower":  # observer beside the touchdown zone (aim point once known)
            spot = info["aim"] if info and "aim" in info else self.runway.center
            cam = np.array([*(spot + self.runway.normal * (self.runway.width / 2 + 25)), -3.0])
            target = pos
        else:  # overhead
            cam = pos + np.array([0.1, 0, -60.0 * z])
            target = pos
        cam[2] = min(cam[2], -1.0)  # stay above ground
        return cam, look_at(cam, target)

    def draw_glider(self, img, cam, cam_pos, R_cam, pos, R_nb):
        shadows, parts = [], []
        for verts, col in self.model:
            w = pos + verts @ R_nb.T
            sh = w.copy()
            sh[:, 2] = 0.0
            shadows.append(sh)
            parts.append((w, col))
        parts.sort(key=lambda p: -np.linalg.norm(p[0].mean(0) - cam_pos))  # painter's algorithm
        overlay = img.copy()
        for sh in shadows:
            uv, ok = cam.world_to_pixel(sh, cam_pos, R_cam)
            if ok.all():
                cv2.fillPoly(overlay, [uv.astype(np.int32)], (30, 40, 30), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        for w, col in parts:
            uv, ok = cam.world_to_pixel(w, cam_pos, R_cam)
            if ok.all():
                pts = uv.astype(np.int32)
                cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
                cv2.polylines(img, [pts], True, (20, 20, 20), 1, cv2.LINE_AA)


def sim_pose(sim):
    return sim.x[0:3].copy(), rot_from_quat(sim.x[6:10])
