"""Omnidirectional camera models and a ray-cast renderer of the ground scene.

Camera frame follows OpenCV: z = optical axis, x = image right, y = image down.

Projections
  fisheye  - single wide lens (the "bulge" eye camera). Radial models:
             equidistant  r = f*theta
             equisolid    r = 2f*sin(theta/2)
             kannala_brandt  r = f*theta_d, theta_d = theta(1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
             (the same model as cv2.fisheye, so a real calibration drops straight in)
  equirect - full-sphere longitude/latitude image (stitched 360 camera output)
  pinhole  - ordinary perspective camera (external/chase views), fov_deg = horizontal FOV
"""
import numpy as np

from .rotations import rot_from_euler

# Camera axes expressed in body axes when the mount angles are all zero:
# optical axis = body x (forward), image right = body y, image down = body z.
_R_BODY_CAM0 = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class OmniCamera:
    def __init__(self, cfg):
        self.projection = cfg.get("projection", "fisheye")
        self.model = cfg.get("model", "equidistant")
        self.kb = np.asarray(cfg.get("kb_coeffs", [0, 0, 0, 0]), dtype=float)
        self.W, self.H = int(cfg["width_px"]), int(cfg["height_px"])
        self.fov = np.radians(cfg.get("fov_deg", 200.0))
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        rpy = np.radians(cfg.get("mount_rpy_deg", [0, -90, 0]))
        self.R_body_cam = rot_from_euler(*rpy) @ _R_BODY_CAM0
        self.offset_body = np.asarray(cfg.get("mount_offset_body_m", [0, 0, 0]), dtype=float)
        self.noise_sigma = float(cfg.get("noise_sigma", 0.0))
        self.visibility = float(cfg.get("visibility_m", 15000.0))

        if self.projection == "fisheye":
            r_max = min(self.W, self.H) / 2
            self.f = r_max / self._radial(self.fov / 2, 1.0)
        elif self.projection == "pinhole":
            self.f = (self.W / 2) / np.tan(self.fov / 2)
        self.rays, self.valid = self._make_rays()
        self.pixel_angle = self._pixel_angle()

    # --- radial model -------------------------------------------------------
    def _radial(self, theta, f):
        if self.model == "equidistant":
            return f * theta
        if self.model == "equisolid":
            return 2 * f * np.sin(theta / 2)
        if self.model == "kannala_brandt":
            t2 = theta * theta
            k1, k2, k3, k4 = self.kb
            return f * theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        raise ValueError(self.model)

    def _theta_from_r(self, r):
        if self.model == "equidistant":
            return r / self.f
        if self.model == "equisolid":
            return 2 * np.arcsin(np.clip(r / (2 * self.f), -1, 1))
        theta = r / self.f  # Newton on the Kannala-Brandt polynomial
        for _ in range(10):
            g = self._radial(theta, self.f) - r
            dg = (self._radial(theta + 1e-6, self.f) - self._radial(theta - 1e-6, self.f)) / 2e-6
            theta = theta - g / dg
        return theta

    # --- rays -----------------------------------------------------------------
    def _make_rays(self):
        u, v = np.meshgrid(np.arange(self.W, dtype=float), np.arange(self.H, dtype=float))
        if self.projection == "fisheye":
            dx, dy = u - self.cx, v - self.cy
            r = np.hypot(dx, dy)
            theta = self._theta_from_r(r)
            phi = np.arctan2(dy, dx)
            valid = (r <= self._radial(self.fov / 2, self.f) + 0.5) & (theta <= self.fov / 2)
            st = np.sin(theta)
            rays = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], -1)
        elif self.projection == "pinhole":
            rays = np.stack([(u - self.cx) / self.f, (v - self.cy) / self.f, np.ones_like(u)], -1)
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            valid = np.ones_like(u, bool)
        elif self.projection == "equirect":
            lon = (u + 0.5) / self.W * 2 * np.pi - np.pi
            lat = np.pi / 2 - (v + 0.5) / self.H * np.pi
            rays = np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
            valid = np.ones_like(lon, bool)
        else:
            raise ValueError(self.projection)
        return rays.reshape(-1, 3), valid.reshape(-1)

    def _pixel_angle(self):
        if self.projection == "equirect":
            return 2 * np.pi / self.W
        return 1.0 / self.f

    # --- projection -----------------------------------------------------------
    def project(self, p_cam):
        """Camera-frame points (N,3) -> pixels (N,2) and in-image mask."""
        p = np.atleast_2d(p_cam).astype(float)
        n = np.linalg.norm(p, axis=1)
        if self.projection == "fisheye":
            theta = np.arccos(np.clip(p[:, 2] / np.maximum(n, 1e-12), -1, 1))
            phi = np.arctan2(p[:, 1], p[:, 0])
            r = self._radial(theta, self.f)
            uv = np.stack([self.cx + r * np.cos(phi), self.cy + r * np.sin(phi)], 1)
            ok = theta <= self.fov / 2
        elif self.projection == "pinhole":
            z = np.where(p[:, 2] > 1e-6, p[:, 2], 1e-6)
            uv = np.stack([self.cx + self.f * p[:, 0] / z, self.cy + self.f * p[:, 1] / z], 1)
            ok = p[:, 2] > 1e-6
        else:
            lon = np.arctan2(p[:, 0], p[:, 2])
            lat = np.arcsin(np.clip(-p[:, 1] / np.maximum(n, 1e-12), -1, 1))
            uv = np.stack([(lon + np.pi) / (2 * np.pi) * self.W - 0.5, (np.pi / 2 - lat) / np.pi * self.H - 0.5], 1)
            ok = np.ones(len(p), bool)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        return uv, ok

    def pose(self, pos_ned, R_ned_body):
        """Camera position (NED) and rotation R_ned_cam for a given aircraft pose."""
        return pos_ned + R_ned_body @ self.offset_body, R_ned_body @ self.R_body_cam

    def world_to_pixel(self, pts_ned, cam_pos, R_ned_cam):
        p_cam = (np.atleast_2d(pts_ned) - cam_pos) @ R_ned_cam
        return self.project(p_cam)


class Renderer:
    """Ray-casts each camera pixel onto the flat ground plane z = 0 (NED)."""

    SKY_ZENITH = np.array([165.0, 110.0, 55.0], np.float32)   # BGR
    SKY_HORIZON = np.array([230.0, 215.0, 200.0], np.float32)
    HAZE = np.array([215.0, 205.0, 195.0], np.float32)

    def __init__(self, camera: OmniCamera, world, seed=0):
        self.cam = camera
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.idx = np.flatnonzero(camera.valid)
        self.rays = camera.rays[self.idx].astype(np.float32)
        if camera.noise_sigma > 0:  # one noise buffer, sampled at a random offset per frame
            n = self.idx.size
            self.noise = self.rng.normal(0, camera.noise_sigma, (n + 65536, 3)).astype(np.float32)

    def render(self, cam_pos, R_ned_cam):
        cam = self.cam
        d = self.rays @ R_ned_cam.T.astype(np.float32)
        dz = d[:, 2]
        out = np.empty((self.idx.size, 3), np.float32)

        ground = dz > 1e-4
        dg = d[ground]
        t = np.float32(-cam_pos[2]) / dg[:, 2]
        hit = cam_pos[None, :2].astype(np.float32) + t[:, None] * dg[:, :2]
        footprint = t * np.float32(cam.pixel_angle) / np.sqrt(dg[:, 2])
        col = self.world.sample(hit, footprint).astype(np.float32)
        fog = np.exp(-t / np.float32(cam.visibility))[:, None]
        out[ground] = col * fog + self.HAZE * (1 - fog)

        sky = ~ground
        w = np.sqrt(np.arcsin(np.clip(-dz[sky], 0, 1)) / np.float32(np.pi / 2))[:, None]
        out[sky] = self.SKY_HORIZON * (1 - w) + self.SKY_ZENITH * w

        if cam.noise_sigma > 0:
            k = int(self.rng.integers(0, 65536))
            out += self.noise[k:k + out.shape[0]]
        img = np.zeros((cam.W * cam.H, 3), np.uint8)
        img[self.idx] = np.clip(out, 0, 255).astype(np.uint8)
        return img.reshape(cam.H, cam.W, 3)
