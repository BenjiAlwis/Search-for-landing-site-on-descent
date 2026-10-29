import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glider_sim.aircraft import AircraftParams, aero_forces_moments  # noqa: E402
from glider_sim.atmosphere import isa  # noqa: E402
from glider_sim.camera import OmniCamera  # noqa: E402
from glider_sim.rotations import euler_from_quat, quat_from_euler, rot_from_euler  # noqa: E402
from glider_sim.sim import Simulation, load_config  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "config" / "default_glider.yaml"


def test_isa_reference_values():
    rho, p, t, a = isa(0.0)
    assert rho == pytest.approx(1.225, rel=1e-3)
    assert t == pytest.approx(288.15)
    rho, p, t, _ = isa(11000.0)
    assert t == pytest.approx(216.77, abs=0.2)  # geometric 11 km is just below the tropopause
    assert p == pytest.approx(22700, rel=0.01)


def test_euler_quat_roundtrip():
    e = np.array([0.3, -0.4, 2.0])
    assert np.allclose(euler_from_quat(quat_from_euler(*e)), e)


@pytest.mark.parametrize("projection,model", [("fisheye", "equidistant"), ("fisheye", "equisolid"),
                                              ("fisheye", "kannala_brandt"), ("equirect", "equidistant")])
def test_camera_project_unproject(projection, model):
    cfg = load_config(CFG)["camera"]
    cfg.update(projection=projection, model=model, width_px=200, height_px=200, kb_coeffs=[0.02, -0.01, 0, 0])
    cam = OmniCamera(cfg)
    idx = np.flatnonzero(cam.valid)[::97]
    uv, ok = cam.project(cam.rays[idx])
    expected = np.stack([idx % cam.W, idx // cam.W], 1)
    assert np.abs(uv[ok] - expected[ok]).max() < 1e-3


def test_steady_glide_matches_polar():
    """Trimmed glide ratio from the full aero model ~ CL/CD of the polar."""
    cfg = load_config(CFG)
    p = AircraftParams.from_config(cfg["aircraft"])
    a = p.aero
    va, rho = 12.0, 1.225
    CL = p.mass * 9.80665 / (0.5 * rho * va ** 2 * p.S)
    alpha = (CL - a["CL0"]) / a["CL_alpha"]
    de = -(a["Cm0"] + a["Cm_alpha"] * alpha) / a["Cm_de"]
    alpha -= a["CL_de"] * de / a["CL_alpha"]
    v = va * np.array([np.cos(alpha), 0, np.sin(alpha)])
    surf = {"elevator": de, "aileron": 0.0, "rudder": 0.0, "flap": 0.0}
    f, m, _ = aero_forces_moments(p, v, np.zeros(3), surf, rho, 1000.0)
    lift = -f[2] * np.cos(alpha) + f[0] * np.sin(alpha)
    drag = -f[0] * np.cos(alpha) - f[2] * np.sin(alpha)
    assert lift == pytest.approx(p.mass * 9.80665, rel=0.03)
    assert abs(m[1]) < 0.02
    assert lift / drag == pytest.approx(1 / p.glide_slope_table(va)[1][0], rel=0.05)


def test_oracle_landing_on_runway():
    cfg = load_config(CFG, {"perception.mode": "oracle", "camera.enabled": False})
    res = Simulation(cfg).run(verbose=False)
    td = res.touchdown
    assert td["on_runway"]
    assert td["sink_rate_mps"] < 1.5
    assert abs(td["cross_track_m"]) < 5
