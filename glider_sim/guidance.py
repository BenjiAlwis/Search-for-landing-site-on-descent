"""Guidance: runway tracking, approach planning and energy management.

Phases
  SEARCH  no runway lock yet: orbit (or fly toward an optional prior position)
  ORBIT   lock acquired: circle an energy-management orbit tangent to the
          extended centreline at the final approach fix (FAF)
  FINAL   straight-in along the centreline to the aim point
  FLARE   exponential sink-rate decay to touchdown

Glide-path control (gliders cannot climb, so the problem is bleeding energy):
the distance-to-go S along the planned path (rest of orbit + n extra orbits +
final) gives a required slope tan(gamma) = h / S. Flaps are scheduled from the
aircraft's own glide polar to fly that slope; elevator holds airspeed. The
number of extra orbits n is the smallest that keeps the required slope within
what full flap can achieve.
"""
from dataclasses import dataclass

import numpy as np

from .rotations import wrap_pi


@dataclass
class RunwayEstimate:
    center: np.ndarray
    heading: float          # axis heading mod pi
    length: float
    width: float
    hits: int = 1


class RunwayTracker:
    """Fuses per-frame detections: gating, axis-ambiguity handling, smoothing."""

    def __init__(self, lock_hits=3, gate_m=60.0, gate_deg=12.0, alpha=0.25):
        self.est = None
        self.lock_hits = lock_hits
        self.gate_m, self.gate = gate_m, np.radians(gate_deg)
        self.alpha = alpha
        self.misses = 0

    @property
    def locked(self):
        return self.est is not None and self.est.hits >= self.lock_hits

    def update(self, det):
        if det is None:
            return self.est
        if self.est is None:
            self.est = RunwayEstimate(det.center_ne.copy(), det.axis_heading, det.length, det.width)
            return self.est
        e = self.est
        dh = wrap_pi(2 * (det.axis_heading - e.heading)) / 2  # difference mod pi
        if np.linalg.norm(det.center_ne - e.center) > self.gate_m or abs(dh) > self.gate:
            self.misses += 1
            if self.misses > 6 and not self.locked:  # restart on a persistent other candidate
                self.est = None
                self.misses = 0
            return self.est
        self.misses = 0
        a = self.alpha
        e.center = (1 - a) * e.center + a * det.center_ne
        e.heading = (e.heading + a * dh) % np.pi
        e.length = (1 - a) * e.length + a * det.length
        e.width = (1 - a) * e.width + a * det.width
        e.hits += 1
        return e


class Guidance:
    def __init__(self, cfg, params, autopilot):
        self.c = cfg
        self.Va = cfg["airspeed_mps"]
        self.R = cfg["orbit_radius_m"]
        self.lam = 1.0 if cfg.get("orbit_direction", "right") == "right" else -1.0
        self.tracker = RunwayTracker()
        self.phase = "SEARCH"
        self.landing_dir = None
        self.search_center = None
        self.prior = cfg.get("search_toward_ne")
        f, tan_g = params.glide_slope_table(self.Va)
        self.flap_f, self.flap_tan = f, tan_g
        self.tan_min, self.tan_max = float(tan_g[0]), float(tan_g[-1])
        self.ap = autopilot
        self.flare_theta0 = None
        self.flare_int = 0.0
        self.wind = np.zeros(2)
        self.info = {}

    # ---------------------------------------------------------------- geometry
    def _geometry(self, est):
        axis = np.array([np.cos(est.heading), np.sin(est.heading)])
        u = self.landing_dir * axis
        thr = est.center - u * est.length / 2
        aim = thr + u * self.c["aim_point_m"]
        faf = aim - u * self.c["final_length_m"]
        right = np.array([-u[1], u[0]])
        center = faf + self.lam * self.R * right
        return u, thr, aim, faf, center

    def _choose_direction(self, est, nav):
        axis = np.array([np.cos(est.heading), np.sin(est.heading)])
        along = self.wind @ axis
        if abs(along) > 0.5:
            self.landing_dir = 1 if along < 0 else -1  # land into wind
        elif self.landing_dir is None:
            to_c = est.center - nav.pos[:2]
            self.landing_dir = 1 if to_c @ axis > 0 else -1  # keep going the same way

    def _orbit_arc(self, p, center, faf):
        phi = np.arctan2(p[1] - center[1], p[0] - center[0])
        phi_exit = np.arctan2(faf[1] - center[1], faf[0] - center[0])
        return (self.lam * (phi_exit - phi)) % (2 * np.pi), phi

    # ---------------------------------------------------------------- path following
    def _follow_orbit(self, p, center, radius, lam, k=2.0):
        d = np.linalg.norm(p - center)
        phi = np.arctan2(p[1] - center[1], p[0] - center[0])
        return phi + lam * (np.pi / 2 + np.arctan(k * (d - radius) / radius))

    @staticmethod
    def _follow_line(p, origin, u, k=0.03, chi_inf=np.radians(70)):
        chi_q = np.arctan2(u[1], u[0])
        e_py = -np.sin(chi_q) * (p[0] - origin[0]) + np.cos(chi_q) * (p[1] - origin[1])
        return chi_q - chi_inf * 2 / np.pi * np.arctan(k * e_py), e_py

    def _flap_for_slope(self, tan_req_ground, gs_ratio):
        tan_air = tan_req_ground * gs_ratio
        return float(np.interp(tan_air, self.flap_tan, self.flap_f))

    # ---------------------------------------------------------------- main update
    def update(self, nav, detection, dt):
        est = self.tracker.update(detection) if detection is not None else self.tracker.est
        p, h = nav.pos[:2], nav.altitude
        cmd = {"airspeed": self.Va, "flap": 0.0}
        if abs(nav.euler[0]) < np.radians(40) and abs(nav.airspeed - self.Va) < 3:  # skip tumbling
            self.wind += (nav.wind_estimate() - self.wind) * min(dt / 20.0, 1.0)

        if self.phase == "SEARCH":
            if self.search_center is None:
                self.search_center = p.copy()
            if self.tracker.locked:
                self._choose_direction(est, nav)
                self.phase = "ORBIT"
            elif self.prior is not None and np.linalg.norm(np.asarray(self.prior) - p) > 300:
                d = np.asarray(self.prior) - p
                cmd["course"] = np.arctan2(d[1], d[0])
            else:
                cmd["course"] = self._follow_orbit(p, self.search_center, self.c["search_radius_m"], self.lam)
            if self.phase == "SEARCH":
                self.info = {"phase": self.phase}
                return cmd

        u, thr, aim, faf, center = self._geometry(est)
        L = self.c["final_length_m"]
        wind_along = self.wind @ u

        if self.phase == "ORBIT" and h > 120:  # wind estimate matures while high
            self._choose_direction(est, nav)
            u, thr, aim, faf, center = self._geometry(est)

        if self.phase == "ORBIT":
            arc, _ = self._orbit_arc(p, center, faf)
            if arc > np.radians(340):  # just passed the exit point: slightly negative arc
                arc -= 2 * np.pi
            d_c = np.linalg.norm(p - center)
            S0 = max(d_c - self.R, 0.0) + self.R * max(arc, 0.0) + L
            n = 0
            while h / (S0 + n * 2 * np.pi * self.R) > 0.8 * self.tan_max and n < 50:
                n += 1
            S = S0 + n * 2 * np.pi * self.R
            tan_req = h / S
            cmd["course"] = self._follow_orbit(p, center, self.R, self.lam)
            cmd["flap"] = self._flap_for_slope(tan_req, 1.0)
            too_low = n == 0 and tan_req < 1.05 * self.tan_min
            at_exit = n == 0 and abs(arc) < np.radians(12) and d_c < 1.6 * self.R
            if at_exit or too_low:
                self.phase = "FINAL"
                self.ap.reset_integrators()
            self.info = {"phase": "ORBIT", "orbits_left": n, "S": S, "tan_req": tan_req}

        if self.phase == "FINAL":
            chi, e_py = self._follow_line(p, faf, u)
            S = max((aim - p) @ u, 1.0)
            tan_req = h / S
            vg_along = max(self.Va + wind_along, 2.0)
            cmd["course"] = chi
            cmd["flap"] = self._flap_for_slope(tan_req, vg_along / self.Va)
            cmd["max_bank"] = np.radians(25 if h > 15 else 10)
            sink = max(nav.vel[2], 0.0)
            if h < max(self.c["flare_height_m"], self.c["flare_tau_s"] * sink):
                self.phase = "FLARE"
                self.flare_theta0 = nav.euler[1]
                self.flare_h0 = h
                self.flare_sink0 = max(sink, 0.5)
                self.flare_flap = cmd["flap"]
            self.info = {"phase": "FINAL", "S": S, "tan_req": tan_req, "cross_track": e_py}

        if self.phase == "FLARE":
            # Sink rate decays linearly with height from its entry value to a gentle touchdown
            chi, e_py = self._follow_line(p, faf, u, k=0.1)
            hdot = -nav.vel[2]
            td_sink = self.c.get("touchdown_sink_mps", 0.3)
            frac = np.clip(h / self.flare_h0, 0.0, 1.0)
            hdot_c = -(td_sink + (self.flare_sink0 - td_sink) * frac)
            e = hdot_c - hdot
            self.flare_int += e * dt
            cmd["course"] = chi                                 # wing-low into wind
            cmd["max_bank"] = np.radians(8)
            cmd["heading"] = float(np.arctan2(u[1], u[0]))      # rudder de-crab
            cmd["flap"] = self.flare_flap
            cmd["theta"] = float(np.clip(self.flare_theta0 + 0.2 * e + 0.4 * self.flare_int,
                                         np.radians(-5), np.radians(14)))
            self.info = {"phase": "FLARE", "hdot_cmd": hdot_c, "cross_track": e_py}

        self.info.update({"runway_center": est.center.copy(), "runway_heading": est.heading,
                          "landing_dir": self.landing_dir, "aim": aim, "faf": faf, "orbit_center": center})
        return cmd
