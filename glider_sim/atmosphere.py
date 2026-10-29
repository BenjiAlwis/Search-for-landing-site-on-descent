"""US Standard Atmosphere 1976 (0-47 km) plus steady wind and Dryden turbulence."""
import numpy as np

G0 = 9.80665
R_AIR = 287.05287
GAMMA = 1.4

# (base geopotential altitude m, base temperature K, lapse rate K/m, base pressure Pa)
_LAYERS = [
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
]
_R_EARTH = 6356766.0


def isa(altitude_msl):
    """Return (density kg/m^3, pressure Pa, temperature K, speed of sound m/s)."""
    h = _R_EARTH * altitude_msl / (_R_EARTH + altitude_msl)  # geopotential
    h = float(np.clip(h, -500.0, 50000.0))
    for i in range(len(_LAYERS) - 1, -1, -1):
        hb, tb, lapse, pb = _LAYERS[i]
        if h >= hb or i == 0:
            break
    t = tb + lapse * (h - hb)
    if lapse == 0.0:
        p = pb * np.exp(-G0 * (h - hb) / (R_AIR * tb))
    else:
        p = pb * (t / tb) ** (-G0 / (lapse * R_AIR))
    rho = p / (R_AIR * t)
    return rho, p, t, np.sqrt(GAMMA * R_AIR * t)


# MIL-F-8785C wind speed at 20 ft (knots) defining each turbulence level
_W20_KT = {"none": 0.0, "light": 15.0, "moderate": 30.0, "severe": 45.0}
_FT = 0.3048
_KT = 0.514444


class Wind:
    """Steady wind plus Dryden turbulence (MIL-F-8785C low-altitude form).

    Gusts are generated in body axes with first-order shaping filters (exact
    for u, the usual first-order approximation for v and w). Valid < 1000 ft
    AGL; above that the low-altitude intensities are held constant.
    """

    def __init__(self, steady_ned, turbulence="none", seed=0):
        self.steady = np.asarray(steady_ned, dtype=float)
        self.sigma_w = 0.1 * _W20_KT[turbulence] * _KT
        self.rng = np.random.default_rng(seed)
        self.gust_body = np.zeros(3)

    def step(self, dt, altitude_agl, airspeed):
        if self.sigma_w <= 0.0:
            return self.gust_body
        h_ft = float(np.clip(altitude_agl / _FT, 10.0, 1000.0))
        k = 0.177 + 0.000823 * h_ft
        lu = lv = h_ft / k ** 1.2 * _FT
        lw = h_ft * _FT
        sw = self.sigma_w
        su = sv = sw / k ** 0.4
        va = max(airspeed, 1.0)
        for i, (sig, length) in enumerate(((su, lu), (sv, lv), (sw, lw))):
            a = np.exp(-va * dt / length)
            self.gust_body[i] = a * self.gust_body[i] + sig * np.sqrt(1 - a * a) * self.rng.standard_normal()
        return self.gust_body
