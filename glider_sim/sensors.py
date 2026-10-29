"""Navigation estimate (what an IMU/GNSS/baro/pitot stack would hand guidance).

Truth plus white noise and slowly wandering biases. Guidance and the vision
pipeline only ever see this estimate, never the true state.
"""
from dataclasses import dataclass

import numpy as np

from .rotations import euler_from_quat, rot_from_euler, rot_from_quat


@dataclass
class NavEstimate:
    t: float
    pos: np.ndarray        # NED [m]
    vel: np.ndarray        # NED [m/s]
    euler: np.ndarray      # roll, pitch, yaw [rad]
    omega: np.ndarray      # body rates [rad/s]
    airspeed: float        # [m/s]

    @property
    def R(self):
        return rot_from_euler(*self.euler)

    @property
    def altitude(self):
        return -self.pos[2]

    @property
    def course(self):
        return float(np.arctan2(self.vel[1], self.vel[0]))

    @property
    def groundspeed(self):
        return float(np.hypot(self.vel[0], self.vel[1]))

    def wind_estimate(self):
        """Horizontal wind = ground velocity - air velocity along the nose."""
        air = self.R @ np.array([self.airspeed, 0.0, 0.0])
        return (self.vel - air)[:2]


class Sensors:
    def __init__(self, cfg, seed=0):
        self.c = cfg
        self.rng = np.random.default_rng(seed + 101)
        self.pos_bias = np.zeros(3)
        self.att_bias = self.rng.normal(0, np.radians(cfg["attitude_bias_deg"]), 3)

    def measure(self, t, x, airspeed, dt):
        c, n = self.c, self.rng.normal
        self.pos_bias[:2] += n(0, c["position_bias_walk_m"] * np.sqrt(dt), 2)
        pos = x[0:3] + self.pos_bias
        pos[:2] += n(0, c["position_sigma_m"], 2)
        pos[2] = x[2] + n(0, c["altitude_sigma_m"])  # baro/laser altimeter, no GNSS bias
        vel = rot_from_quat(x[6:10]) @ x[3:6] + n(0, c["velocity_sigma_mps"], 3)
        euler = euler_from_quat(x[6:10]) + self.att_bias + n(0, np.radians(c["attitude_sigma_deg"]), 3)
        omega = x[10:13] + n(0, np.radians(c["rate_sigma_dps"]), 3)
        va = airspeed + n(0, c["airspeed_sigma_mps"])
        return NavEstimate(t, pos, vel, euler, omega, float(va))
