"""Glider parameters, aerodynamics, actuators and 6-DOF rigid-body dynamics.

Aerodynamic model: standard stability-derivative build-up (Beard & McLain,
"Small Unmanned Aircraft", ch. 4) with
  * parabolic drag polar  CD = CD0 + K * CL^2  (K = 1 / (pi e AR))
  * sigmoid blend to flat-plate lift past stall
  * McCormick ground-effect reduction of induced drag
  * flap lift / drag / pitching-moment increments
"""
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import G0
from .rotations import quat_derivative, rot_from_quat

CHANNELS = ("elevator", "aileron", "rudder", "flap")


@dataclass
class AircraftParams:
    mass: float
    J: np.ndarray
    S: float
    b: float
    c: float
    e: float
    gear_height: float
    aero: dict
    mixer: str
    actuators: dict
    name: str = "glider"
    AR: float = field(init=False)
    K: float = field(init=False)

    def __post_init__(self):
        self.AR = self.b ** 2 / self.S
        self.K = 1.0 / (np.pi * self.e * self.AR)
        self.J_inv = np.linalg.inv(self.J)

    @classmethod
    def from_config(cls, cfg):
        jx, jy, jz, jxz = cfg["Jx"], cfg["Jy"], cfg["Jz"], cfg["Jxz"]
        J = np.array([[jx, 0, -jxz], [0, jy, 0], [-jxz, 0, jz]], dtype=float)
        aero = dict(cfg["aero"])
        aero["alpha_stall"] = np.radians(aero.pop("alpha_stall_deg"))
        return cls(
            mass=cfg["mass_kg"], J=J, S=cfg["wing_area_m2"], b=cfg["span_m"],
            c=cfg["chord_m"], e=cfg["oswald_e"], gear_height=cfg.get("gear_height_m", 0.0),
            aero=aero, mixer=cfg.get("mixer", "conventional"), actuators=cfg["actuators"],
            name=cfg.get("name", "glider"),
        )

    def glide_slope_table(self, airspeed, rho=1.225, n=11):
        """Steady-glide tan(gamma) vs flap fraction at a given airspeed (still air)."""
        a = self.aero
        f = np.linspace(0, 1, n)
        df = np.radians(self.actuators["flap"].get("min_deg", 0)) + f * np.radians(
            self.actuators["flap"]["max_deg"] - self.actuators["flap"].get("min_deg", 0))
        CL = self.mass * G0 / (0.5 * rho * airspeed ** 2 * self.S)
        CD = a["CD0"] + a["CD_df"] * np.abs(df) + self.K * CL ** 2
        return f, CD / CL


# ----------------------------------------------------------------------------
# Actuators and mixing
# ----------------------------------------------------------------------------
class Actuators:
    """Maps virtual commands (elevator/aileron/rudder/flap, radians) through the
    surface mixer to physical servos with lag, rate and position limits, then
    back to the effective virtual deflections the aero model uses."""

    def __init__(self, params: AircraftParams):
        self.mixer = params.mixer
        acts = params.actuators
        if self.mixer == "conventional":
            self.servos = {ch: acts[ch] for ch in CHANNELS}
        elif self.mixer == "elevon":
            self.servos = {"elevon_left": acts["elevator"], "elevon_right": acts["elevator"],
                           "rudder": acts["rudder"], "flap": acts["flap"]}
        elif self.mixer == "vtail":
            self.servos = {"ruddervator_left": acts["elevator"], "ruddervator_right": acts["elevator"],
                           "aileron": acts["aileron"], "flap": acts["flap"]}
        else:
            raise ValueError(f"unknown mixer {self.mixer}")
        self.pos = {k: 0.0 for k in self.servos}

    def _mix(self, cmd):
        de, da, dr, df = (cmd.get(ch, 0.0) for ch in CHANNELS)
        if self.mixer == "conventional":
            return dict(cmd)
        if self.mixer == "elevon":
            return {"elevon_left": de + da, "elevon_right": de - da, "rudder": dr, "flap": df}
        return {"ruddervator_left": de - dr, "ruddervator_right": de + dr, "aileron": da, "flap": df}

    def _demix(self, pos):
        if self.mixer == "conventional":
            return dict(pos)
        if self.mixer == "elevon":
            l, r = pos["elevon_left"], pos["elevon_right"]
            return {"elevator": (l + r) / 2, "aileron": (l - r) / 2, "rudder": pos["rudder"], "flap": pos["flap"]}
        l, r = pos["ruddervator_left"], pos["ruddervator_right"]
        return {"elevator": (l + r) / 2, "rudder": (r - l) / 2, "aileron": pos["aileron"], "flap": pos["flap"]}

    def step(self, cmd, dt):
        for name, target in self._mix(cmd).items():
            s = self.servos[name]
            lo = np.radians(s.get("min_deg", -s["max_deg"]))
            hi = np.radians(s["max_deg"])
            target = np.clip(target, lo, hi)
            rate = (target - self.pos[name]) / max(s["tau_s"], dt)
            rate = np.clip(rate, -np.radians(s["rate_dps"]), np.radians(s["rate_dps"]))
            self.pos[name] = float(np.clip(self.pos[name] + rate * dt, lo, hi))
        return self.effective()

    def effective(self):
        return self._demix(self.pos)


# ----------------------------------------------------------------------------
# Aerodynamics
# ----------------------------------------------------------------------------
def air_data(v_rel_body):
    u, v, w = v_rel_body
    va = float(np.linalg.norm(v_rel_body))
    if va < 1e-3:
        return va, 0.0, 0.0
    return va, float(np.arctan2(w, u)), float(np.arcsin(np.clip(v / va, -1, 1)))


def aero_forces_moments(params: AircraftParams, v_rel_body, omega, surf, rho, h_agl):
    a = params.aero
    va, alpha, beta = air_data(v_rel_body)
    if va < 0.5:
        return np.zeros(3), np.zeros(3), (va, alpha, beta)
    p, q, r = omega
    de, da, dr, df = (surf[ch] for ch in CHANNELS)
    qbar = 0.5 * rho * va * va
    S, b, c = params.S, params.b, params.c
    cq = c / (2 * va)
    bq = b / (2 * va)

    # Lift with stall blending
    a0, M = a["alpha_stall"], a["stall_blend_M"]
    sig = (1 + np.exp(-M * (alpha - a0)) + np.exp(M * (alpha + a0))) / (
        (1 + np.exp(-M * (alpha - a0))) * (1 + np.exp(M * (alpha + a0))))
    cl_lin = a["CL0"] + a["CL_alpha"] * alpha
    cl_plate = 2 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    CL = (1 - sig) * cl_lin + sig * cl_plate + a["CL_q"] * cq * q + a["CL_de"] * de + a["CL_df"] * df

    # Drag: parabolic polar (with ground effect on the induced part) + flat-plate post-stall
    k = params.K
    if a.get("ground_effect", False):
        x = 16.0 * max(h_agl, 0.0) / b
        k *= x * x / (1 + x * x)
    CD = a["CD0"] + k * CL ** 2 + a["CD_df"] * abs(df) + a.get("CD_de", 0.0) * abs(de)
    CD += sig * 1.2 * np.sin(alpha) ** 2

    CY = a["CY_beta"] * beta + a["CY_p"] * bq * p + a["CY_r"] * bq * r + a["CY_da"] * da + a["CY_dr"] * dr
    Cl = a["Cl_beta"] * beta + a["Cl_p"] * bq * p + a["Cl_r"] * bq * r + a["Cl_da"] * da + a["Cl_dr"] * dr
    Cm = a["Cm0"] + a["Cm_alpha"] * alpha + a["Cm_q"] * cq * q + a["Cm_de"] * de + a["Cm_df"] * df
    Cn = a["Cn_beta"] * beta + a["Cn_p"] * bq * p + a["Cn_r"] * bq * r + a["Cn_da"] * da + a["Cn_dr"] * dr

    ca, sa = np.cos(alpha), np.sin(alpha)
    lift, drag = qbar * S * CL, qbar * S * CD
    force = np.array([-drag * ca + lift * sa, qbar * S * CY, -drag * sa - lift * ca])
    moment = qbar * S * np.array([b * Cl, c * Cm, b * Cn])
    return force, moment, (va, alpha, beta)


# ----------------------------------------------------------------------------
# Rigid-body dynamics
# ----------------------------------------------------------------------------
# state = [pn, pe, pd, u, v, w, q0, q1, q2, q3, p, q, r]
def derivatives(x, params: AircraftParams, surf, rho, wind_ned, gust_body):
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]
    R = rot_from_quat(quat)
    v_rel = vel - R.T @ wind_ned - gust_body
    f_aero, m_aero, _ = aero_forces_moments(params, v_rel, omega, surf, rho, -x[2])
    f_grav = R.T @ np.array([0.0, 0.0, params.mass * G0])

    dx = np.empty(13)
    dx[0:3] = R @ vel
    dx[3:6] = (f_aero + f_grav) / params.mass - np.cross(omega, vel)
    dx[6:10] = quat_derivative(quat, omega)
    dx[10:13] = params.J_inv @ (m_aero - np.cross(omega, params.J @ omega))
    return dx


def rk4_step(x, dt, *args):
    k1 = derivatives(x, *args)
    k2 = derivatives(x + 0.5 * dt * k1, *args)
    k3 = derivatives(x + 0.5 * dt * k2, *args)
    k4 = derivatives(x + dt * k3, *args)
    x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    x[6:10] /= np.linalg.norm(x[6:10])
    return x
