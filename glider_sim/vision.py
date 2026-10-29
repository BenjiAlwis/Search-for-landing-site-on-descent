"""Runway perception.

`RunwayDetector` is a classical baseline meant to be replaced by a learned
model. It:
  1. re-projects the fisheye image onto the ground plane (bird's-eye view)
     using the *estimated* camera pose (nav attitude + altitude),
  2. segments low-saturation asphalt / white paint,
  3. fits rotated rectangles and keeps the most runway-like one.

Any replacement only needs to return a `RunwayDetection` (or None) from
`detect(image, nav)`.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RunwayDetection:
    center_ne: np.ndarray      # runway centre in the nav frame [m]
    axis_heading: float        # runway axis direction [rad], ambiguous by pi
    length: float
    width: float
    score: float
    debug: dict = field(default_factory=dict)


class RunwayDetector:
    def __init__(self, camera, bev_px=1024, min_len=80.0, max_len=4000.0, min_wid=8.0, max_wid=90.0):
        self.cam = camera
        self.n = bev_px
        self.min_len, self.max_len = min_len, max_len
        self.min_wid, self.max_wid = min_wid, max_wid
        self.last_debug = None

    def bev(self, image, nav):
        """Ground-plane view centred under the aircraft, north up."""
        h = max(nav.altitude, 1.0)
        half = float(np.clip(6.0 * h, 450.0, 2000.0))
        res = 2 * half / self.n
        ax = (np.arange(self.n) + 0.5) * res - half
        east, north = np.meshgrid(ax, -ax)
        pts = np.stack([north.ravel() + nav.pos[0], east.ravel() + nav.pos[1], np.zeros(north.size)], 1)
        cam_pos, R = self.cam.pose(nav.pos, nav.R)
        uv, ok = self.cam.world_to_pixel(pts, cam_pos, R)
        uv[~ok] = -10
        mx = uv[:, 0].reshape(self.n, self.n).astype(np.float32)
        my = uv[:, 1].reshape(self.n, self.n).astype(np.float32)
        img = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, ok.reshape(self.n, self.n), half, res

    def detect(self, image, nav):
        """Returns a RunwayDetection or None. Every intermediate stage is kept in
        `self.last_debug` (also on failure) for visualisation."""
        bev, ok, half, res = self.bev(image, nav)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        asphalt = (s < 40) & (v > 35) & (v < 140) & ok
        paint = (s < 45) & (v > 185) & ok
        mask_raw = (asphalt | paint).astype(np.uint8) * 255
        k = max(3, int(round(4.0 / res)) | 1)
        mask = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        ko = max(2, int(round(self.min_wid * 0.6 / res)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        candidates = []  # (box_px, verdict) for every shape big enough to consider
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area * res * res < self.min_len * self.min_wid * 0.5:
                continue
            (cx, cy), (w, h), _ = rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            length, width = max(w, h) * res, min(w, h) * res
            fill = area / max(w * h, 1e-6)
            if not (self.min_len <= length <= self.max_len and self.min_wid <= width <= self.max_wid):
                verdict = f"size {length:.0f}x{width:.0f}m"
            elif length / width < 4.0:
                verdict = f"aspect {length / width:.1f}"
            elif fill < 0.45:
                verdict = f"fill {fill:.2f}"
            elif (box.min() < 2) or (box.max() > self.n - 3):  # truncated by the view edge
                verdict = "cut by edge"
            else:
                score = fill * min(length / width / 8.0, 1.0)
                verdict = f"ok {score:.2f}"
                if best is None or score > best[0]:
                    best = (score, rect, box, length, width)
            candidates.append((box, verdict))

        self.last_debug = {"bev": bev, "valid": ok, "sat": s, "val": v, "asphalt": asphalt, "paint": paint,
                           "mask_raw": mask_raw, "mask": mask, "candidates": candidates,
                           "box_px": None if best is None else best[2], "half": half, "res": res,
                           "kernels": (k, ko)}
        if best is None:
            return None

        score, ((cx, cy), _, _), box, length, width = best
        e0 = box[1] - box[0]
        e1 = box[2] - box[1]
        long_edge = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
        heading = np.arctan2(long_edge[0], -long_edge[1]) % np.pi  # (east, north) from (col, -row)
        center = nav.pos[:2] + np.array([half - (cy + 0.5) * res, (cx + 0.5) * res - half])
        return RunwayDetection(center, float(heading), float(length), float(width), float(score),
                               debug=self.last_debug)


class OraclePerception:
    """Truth-based detections with noise, for fast guidance development.

    The detection is expressed in the nav frame (so nav position error is
    inherited exactly as it would be with a real camera)."""

    def __init__(self, runway, pos_sigma, heading_sigma_deg, seed=0):
        self.rwy = runway
        self.pos_sigma = pos_sigma
        self.hdg_sigma = np.radians(heading_sigma_deg)
        self.rng = np.random.default_rng(seed + 202)

    def detect(self, true_pos, nav):
        rel = self.rwy.center - true_pos[:2]
        center = nav.pos[:2] + rel + self.rng.normal(0, self.pos_sigma, 2)
        hdg = (self.rwy.heading + self.rng.normal(0, self.hdg_sigma)) % np.pi
        return RunwayDetection(center, float(hdg), self.rwy.length, self.rwy.width, 1.0)


def runway_label(camera, runway, cam_pos, R_ned_cam):
    """Ground-truth runway corners/centre in image pixels, for dataset labels."""
    corners = np.column_stack([runway.corners_ne(), np.zeros(4)])
    center = np.array([[*runway.center, 0.0]])
    uv, ok = camera.world_to_pixel(np.vstack([corners, center]), cam_pos, R_ned_cam)
    return {"corners_px": uv[:4].tolist(), "corners_visible": ok[:4].tolist(),
            "center_px": uv[4].tolist(), "center_visible": bool(ok[4])}
