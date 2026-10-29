"""Closed-loop simulation: physics <-> sensors <-> camera/perception <-> guidance <-> autopilot."""
import copy
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml

from .aircraft import AircraftParams, Actuators, air_data, rk4_step
from .atmosphere import Wind, isa
from .autopilot import Autopilot
from .camera import OmniCamera, Renderer
from .guidance import Guidance
from .rotations import euler_from_quat, quat_from_euler, rot_from_quat
from .sensors import Sensors
from .viewer import put_label
from .vision import OraclePerception, RunwayDetector
from .world import Runway, World


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in (overrides or {}).items():  # "a.b.c" -> value
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = val
    return cfg


@dataclass
class SimResult:
    log: dict
    touchdown: dict = field(default_factory=dict)
    gains: dict = field(default_factory=dict)


class Simulation:
    def __init__(self, cfg, video_path=None, frame_callback=None):
        self.cfg = cfg = copy.deepcopy(cfg)
        seed = cfg["environment"].get("seed", 0)
        self.params = AircraftParams.from_config(cfg["aircraft"])
        self.act = Actuators(self.params)
        self.wind = Wind(cfg["environment"]["wind_ned_mps"], cfg["environment"].get("turbulence", "none"), seed)
        self.ground_msl = cfg["environment"].get("ground_elevation_m", 0.0)
        r = cfg["runway"]
        self.runway = Runway(r["center_ne_m"], r["heading_deg"], r["length_m"], r["width_m"])
        self.sensors = Sensors(cfg["sensors"], seed)

        g = cfg["guidance"]
        rho0 = isa(self.ground_msl)[0]
        self.ap = Autopilot(self.params, g["airspeed_mps"], rho0, g["max_bank_deg"])
        self.guid = Guidance(g, self.params, self.ap)

        self.mode = cfg["perception"]["mode"]
        cam_cfg = cfg["camera"]
        self.use_camera = cam_cfg.get("enabled", True) and (self.mode == "vision" or video_path or frame_callback)
        if self.use_camera:
            self.world = World(self.runway, seed=seed)
            self.cam = OmniCamera(cam_cfg)
            self.renderer = Renderer(self.cam, self.world, seed)
            self.detector = RunwayDetector(self.cam)
        p = cfg["perception"]
        self.oracle = OraclePerception(self.runway, p["oracle_pos_sigma_m"], p["oracle_heading_sigma_deg"], seed)
        self.video_path = video_path
        self.frame_callback = frame_callback

        ic = cfg["initial_state"]
        self.x = np.concatenate([
            np.asarray(ic["position_ned_m"], float),
            np.asarray(ic["velocity_body_mps"], float),
            quat_from_euler(*np.radians(ic["euler_deg"])),
            np.radians(ic["rates_dps"]),
        ])

    # ------------------------------------------------------------------
    def _airspeed(self, x):
        R = rot_from_quat(x[6:10])
        return air_data(x[3:6] - R.T @ self.wind.steady - self.wind.gust_body)

    def run(self, verbose=True):
        sc = self.cfg["sim"]
        dt = sc["dt_s"]
        n_ctrl = max(1, int(round(1.0 / (sc["control_rate_hz"] * dt))))
        dt_ctrl = n_ctrl * dt
        n_cam = max(1, int(round(1.0 / (self.cfg["camera"]["rate_hz"] * dt))))
        t, step = 0.0, 0
        log = {k: [] for k in ("t", "pos", "vel_ned", "euler", "omega", "airspeed", "alpha", "beta",
                               "surf", "cmd_course", "cmd_flap", "phase", "nav_pos", "rwy_est", "wind")}
        writer = None
        detection = None
        cmd = {"course": 0.0, "airspeed": self.guid.Va, "flap": 0.0}
        surf_cmd = {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0, "flap": 0.0}
        last_phase = None
        touchdown = {}
        n_frames = n_det = 0

        while t < sc["max_time_s"]:
            va, alpha, beta = self._airspeed(self.x)
            if step % n_ctrl == 0:
                nav = self.sensors.measure(t, self.x, va, dt_ctrl)
                detection = None
                if step % n_cam == 0 and (self.use_camera or self.mode == "oracle"):
                    detection, frame = self._perceive(nav)
                    n_frames += 1
                    n_det += detection is not None
                    if frame is not None and self.frame_callback:  # raw camera image, no annotations
                        if self.frame_callback(self, frame, nav, detection) is False:
                            break
                    if frame is not None and self.video_path:
                        frame = self._overlay(frame, nav, detection)
                        if writer is None:
                            writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                                     self.cfg["camera"]["rate_hz"], frame.shape[1::-1])
                        writer.write(frame)
                cmd = self.guid.update(nav, detection, dt_ctrl)
                surf_cmd = self.ap.update(nav, cmd, dt_ctrl)
                if verbose and self.guid.phase != last_phase:
                    print(f"t={t:6.1f}s  h={-self.x[2]:6.1f}m  phase -> {self.guid.phase}")
                    last_phase = self.guid.phase
                self._log(log, t, va, alpha, beta, cmd, nav)

            surf = self.act.step(surf_cmd, dt)
            h_agl = -self.x[2]
            rho = isa(self.ground_msl + h_agl)[0]
            gust = self.wind.step(dt, h_agl, va)
            self.x = rk4_step(self.x, dt, self.params, surf, rho, self.wind.steady, gust)
            t += dt
            step += 1

            if -self.x[2] <= self.params.gear_height:
                touchdown = self._touchdown_report(t)
                break
            if not np.all(np.isfinite(self.x)):
                raise FloatingPointError(f"state diverged at t={t:.2f}s")

        if writer is not None:
            writer.release()
        if verbose:
            if self.use_camera and self.mode == "vision":
                print(f"camera frames: {n_frames}, runway detections: {n_det}")
            self._print_touchdown(touchdown, t)
        log = {k: np.array(v, dtype=object if k in ("phase", "rwy_est") else float) for k, v in log.items()}
        return SimResult(log, touchdown, self.ap.gains())

    # ------------------------------------------------------------------
    def _perceive(self, nav):
        if self.mode == "oracle":
            frame = None
            if self.use_camera:
                frame = self.renderer.render(*self.cam.pose(self.x[0:3], rot_from_quat(self.x[6:10])))
            return self.oracle.detect(self.x[0:3], nav), frame
        frame = self.renderer.render(*self.cam.pose(self.x[0:3], rot_from_quat(self.x[6:10])))
        return self.detector.detect(frame, nav), frame

    def _overlay(self, frame, nav, det):
        out = frame.copy()
        if det is not None:
            n_, e_ = np.cos(det.axis_heading), np.sin(det.axis_heading)
            ax, nm = np.array([n_, e_]), np.array([-e_, n_])
            l, w = det.length / 2, det.width / 2
            corners = [det.center_ne + sa * l * ax + sc * w * nm for sa, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            s = np.linspace(0, 1, 25, endpoint=False)[:, None]  # straight edges curve in a fisheye
            pts = np.vstack([corners[i] + s * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])
            uv, ok = self.cam.world_to_pixel(np.column_stack([pts, np.zeros(len(pts))]),
                                             *self.cam.pose(nav.pos, nav.R))
            if ok.all():
                cv2.polylines(out, [uv.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
        info = self.guid.info
        lines = [f"t {nav.t:5.1f}s  h {nav.altitude:6.1f} m  V {nav.airspeed:4.1f} m/s",
                 f"{info.get('phase', '')}  flap {self.act.effective()['flap'] * 57.3:4.0f} deg"]
        if "orbits_left" in info:
            lines.append(f"orbits left {info['orbits_left']}")
        for i, s in enumerate(lines):
            put_label(out, s, (10, 22 + 22 * i))
        return out

    def _log(self, log, t, va, alpha, beta, cmd, nav):
        R = rot_from_quat(self.x[6:10])
        log["t"].append(t)
        log["pos"].append(self.x[0:3].copy())
        log["vel_ned"].append(R @ self.x[3:6])
        log["euler"].append(euler_from_quat(self.x[6:10]))
        log["omega"].append(self.x[10:13].copy())
        log["airspeed"].append(va)
        log["alpha"].append(alpha)
        log["beta"].append(beta)
        s = self.act.effective()
        log["surf"].append([s["elevator"], s["aileron"], s["rudder"], s["flap"]])
        log["cmd_course"].append(cmd.get("course", np.nan))
        log["cmd_flap"].append(cmd.get("flap", 0.0))
        log["phase"].append(self.guid.phase)
        log["nav_pos"].append(nav.pos.copy())
        est = self.guid.tracker.est
        log["rwy_est"].append(None if est is None else (est.center.copy(), est.heading))
        log["wind"].append(nav.wind_estimate())

    def _touchdown_report(self, t):
        x = self.x
        R = rot_from_quat(x[6:10])
        v = R @ x[3:6]
        along, cross = self.runway.to_local(x[0:2])
        d = self.guid.landing_dir or 1
        along *= d
        va, _, _ = self._airspeed(x)
        roll, pitch, yaw = euler_from_quat(x[6:10])
        on_rwy = abs(along) <= self.runway.length / 2 and abs(cross) <= self.runway.width / 2
        return {
            "time_s": t,
            "on_runway": bool(on_rwy),
            "dist_from_threshold_m": float(along + self.runway.length / 2),
            "cross_track_m": float(cross * d),
            "sink_rate_mps": float(v[2]),
            "airspeed_mps": float(va),
            "groundspeed_mps": float(np.hypot(v[0], v[1])),
            "heading_err_deg": float(np.degrees(((yaw - (self.runway.heading + (0 if d > 0 else np.pi))) + np.pi)
                                                % (2 * np.pi) - np.pi)),
            "roll_deg": float(np.degrees(roll)),
            "pitch_deg": float(np.degrees(pitch)),
        }

    @staticmethod
    def _print_touchdown(td, t):
        if not td:
            print(f"no touchdown within {t:.0f}s")
            return
        print("---- touchdown ----")
        for k, v in td.items():
            print(f"  {k:24s} {v:.2f}" if isinstance(v, float) else f"  {k:24s} {v}")
