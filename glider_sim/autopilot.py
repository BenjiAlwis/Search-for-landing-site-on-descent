"""Inner-loop autopilot: course / airspeed / flap commands -> surface deflections.

Gains are derived from the aircraft's own aero derivatives by successive loop
closure (Beard & McLain ch. 6) at the commanded airspeed, so they re-tune
automatically when real glider specs replace the placeholder ones.

  aileron  <- roll attitude PD  <- course PI
  elevator <- pitch attitude PD <- airspeed PI (gliders hold speed with pitch)
  rudder   <- washed-out yaw-rate damper
  flap     <- passed through from guidance (glide-path control)
"""
import numpy as np

from .atmosphere import G0
from .rotations import wrap_pi


class Autopilot:
    def __init__(self, params, airspeed, rho=1.225, max_bank_deg=35.0):
        self.p = params
        self.max_bank = np.radians(max_bank_deg)
        a, J = params.aero, params.J
        Va, S, b, c = airspeed, params.S, params.b, params.c
        qbar = 0.5 * rho * Va * Va
        acts = params.actuators

        # ---- roll ----
        jx, jy, jz, jxz = J[0, 0], J[1, 1], J[2, 2], -J[0, 2]
        gam = jx * jz - jxz * jxz
        g3, g4 = jz / gam, jxz / gam
        cp_p = g3 * a["Cl_p"] + g4 * a["Cn_p"]
        cp_da = g3 * a["Cl_da"] + g4 * a["Cn_da"]
        a_phi1 = -qbar * S * b * cp_p * b / (2 * Va)
        a_phi2 = qbar * S * b * cp_da
        da_max = np.radians(acts["aileron"]["max_deg"]) * 0.8
        self.kp_phi = da_max / np.radians(30) * np.sign(a_phi2)
        wn_phi = np.sqrt(abs(a_phi2) * abs(self.kp_phi))
        self.kd_phi = (2 * 0.9 * wn_phi - a_phi1) / a_phi2

        # ---- course ----
        wn_chi = wn_phi / 12.0
        self.kp_chi = 2 * 1.0 * wn_chi * Va / G0
        self.ki_chi = wn_chi ** 2 * Va / G0 * 0.5

        # ---- pitch ----
        k = rho * Va * Va * c * S / (2 * jy)
        a_th1 = -k * a["Cm_q"] * c / (2 * Va)
        a_th2 = -k * a["Cm_alpha"]
        a_th3 = k * a["Cm_de"]
        de_max = np.radians(acts["elevator"]["max_deg"]) * 0.8
        self.kp_th = de_max / np.radians(15) * np.sign(a_th3)
        wn_th = np.sqrt(a_th2 + abs(self.kp_th * a_th3))
        self.kd_th = (2 * 0.7 * wn_th - a_th1) / a_th3
        k_th_dc = self.kp_th * a_th3 / (a_th2 + self.kp_th * a_th3)

        # ---- airspeed via pitch ----
        CL = params.mass * G0 / (qbar * S)
        CD = a["CD0"] + params.K * CL * CL
        a_v1 = rho * Va * S * CD / params.mass
        wn_v = wn_th / 20.0
        self.kp_v = (a_v1 - 2 * 0.9 * wn_v) / (k_th_dc * G0)
        self.ki_v = -wn_v ** 2 / (k_th_dc * G0)

        # ---- yaw damper ----
        self.k_yd = 0.4 * -np.sign(a["Cn_dr"])

        # ---- trim feed-forward vs flap fraction ----
        self.flap_lo = np.radians(acts["flap"].get("min_deg", 0))
        self.flap_hi = np.radians(acts["flap"]["max_deg"])
        f, tan_g = params.glide_slope_table(Va, rho)
        df = self.flap_lo + f * (self.flap_hi - self.flap_lo)
        alpha = (CL - a["CL0"] - a["CL_df"] * df) / a["CL_alpha"]
        self.ff_f = f
        self.ff_theta = alpha - np.arctan(tan_g)
        self.ff_de = -(a["Cm0"] + a["Cm_alpha"] * alpha + a["Cm_df"] * df) / a["Cm_de"]

        self.int_chi = 0.0
        self.int_v = 0.0
        self.r_lp = 0.0
        self.theta_cmd = 0.0
        self.phi_cmd = 0.0

    def gains(self):
        return {k: float(v) for k, v in vars(self).items() if k.startswith(("kp_", "kd_", "ki_", "k_"))}

    def reset_integrators(self):
        self.int_chi = self.int_v = 0.0

    def update(self, nav, cmd, dt):
        """cmd: dict with course (rad), airspeed (m/s), flap (0..1) and optional
        phi (bank override, rad) / theta (pitch override, rad)."""
        roll, pitch, _ = nav.euler
        p, q, r = nav.omega
        flap = float(np.clip(cmd.get("flap", 0.0), 0, 1))

        # Lateral
        if "phi" in cmd:
            phi_c = cmd["phi"]
        else:
            e_chi = wrap_pi(cmd["course"] - nav.course)
            self.int_chi = np.clip(self.int_chi + e_chi * dt, -0.3 / max(self.ki_chi, 1e-6), 0.3 / max(self.ki_chi, 1e-6))
            phi_c = self.kp_chi * e_chi + self.ki_chi * self.int_chi
        lim = cmd.get("max_bank", self.max_bank)
        self.phi_cmd = phi_c = float(np.clip(phi_c, -lim, lim))
        da = self.kp_phi * (phi_c - roll) - self.kd_phi * p

        # Longitudinal
        theta_ff = np.interp(flap, self.ff_f, self.ff_theta)
        if "theta" in cmd:
            theta_c = cmd["theta"]
        else:
            e_v = cmd["airspeed"] - nav.airspeed
            self.int_v = np.clip(self.int_v + e_v * dt, -0.35 / abs(self.ki_v), 0.35 / abs(self.ki_v))
            theta_c = theta_ff + self.kp_v * e_v + self.ki_v * self.int_v
        self.theta_cmd = theta_c = float(np.clip(theta_c, np.radians(-35), np.radians(20)))
        de = np.interp(flap, self.ff_f, self.ff_de) + self.kp_th * (theta_c - pitch) - self.kd_th * q
        # A banked turn needs extra lift: bias the pitch error nose-up with load factor
        de += self.kp_th * (1 / max(np.cos(roll), 0.5) - 1) * 0.15

        # Yaw damper (washout removes the steady turn rate)
        self.r_lp += (r - self.r_lp) * dt / 1.0
        dr = self.k_yd * (r - self.r_lp)
        if "heading" in cmd:  # de-crab: point the nose with rudder, track with bank
            dr -= self.k_yd * 2.0 * wrap_pi(cmd["heading"] - nav.euler[2])

        df = self.flap_lo + flap * (self.flap_hi - self.flap_lo)
        return {"elevator": float(de), "aileron": float(da), "rudder": float(dr), "flap": float(df)}
