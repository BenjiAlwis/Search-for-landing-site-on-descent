"""Explore the simulated environment in a multi-panel dashboard.

  python scripts/explore.py            # watch the autopilot fly live
  python scripts/explore.py --fast     # smoother: half-res outside view, truth-based perception
  python scripts/explore.py --free     # fly around yourself, no physics
  python scripts/explore.py --record out/explore.mp4               # also save the dashboard as a video
  python scripts/explore.py --no-window --record out/explore.mp4   # record only, no window

Panels: 1 outside  2 eye  3 birds-eye  4 saturation  5 segmentation  6 candidates
        7 map  8 glide analysis  9 timeline

Keys (click the window first)
  all modes   click a tile / 1-9 / Tab   show that panel large
              v  outside camera (chase / tower / side / overhead)    + / -  zoom
              c  run the vision pipeline for display in --fast/--oracle mode
              q / Esc  quit
  live        space  pause / resume
  free        w / s  forward / back       a / d  yaw left / right
              r / f  up / down            i / k  pitch nose down / up
              j / l  roll left / right    shift (W A S D R F) = 5x faster
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glider_sim.aircraft import AircraftParams  # noqa: E402
from glider_sim.camera import OmniCamera  # noqa: E402
from glider_sim.rotations import rot_from_euler  # noqa: E402
from glider_sim.sensors import NavEstimate  # noqa: E402
from glider_sim.sim import Simulation, load_config  # noqa: E402
from glider_sim.viewer import Explorer, sim_pose  # noqa: E402
from glider_sim.vision import RunwayDetector  # noqa: E402
from glider_sim.world import Runway, World  # noqa: E402

WINDOW = "glider environment"


class Recorder:
    def __init__(self, path, fps):
        self.path, self.fps, self.w = path, fps, None

    def write(self, img):
        if not self.path:
            return
        if self.w is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, img.shape[1::-1])
        self.w.write(img)

    def close(self):
        if self.w is not None:
            self.w.release()


def attach_mouse(ex):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ex.handle_click(x, y)
    cv2.setMouseCallback(WINDOW, on_mouse)


def run_live(cfg, rec, window=True, render_scale=1.0, cv_preview=True):
    sim = Simulation(cfg, frame_callback=lambda *a: None)  # callback forces camera rendering
    ex = Explorer(sim.world, sim.params, sim.cam, sim.runway, cfg["environment"].get("seed", 0), render_scale,
                  cfg["guidance"]["airspeed_mps"])
    if window:
        attach_mouse(ex)
    # Frame pacing: show frames at a steady interval (the slowest recent frame time)
    # instead of as-soon-as-ready, which looks jerky when per-frame cost varies.
    state = {"paused": False, "last": time.perf_counter(), "interval": 0.0, "cv": cv_preview}

    def on_frame(s, frame, nav, det):
        pos, R = sim_pose(s)
        info = dict(s.guid.info)
        info["orbit_radius"] = s.guid.R
        debug = None
        if s.mode == "vision":
            debug = s.detector.last_debug  # the detector already ran on this frame
        elif state["cv"]:
            s.detector.detect(frame, nav)   # display only; guidance still uses the oracle
            debug = s.detector.last_debug
        flap = s.act.effective()["flap"] / np.radians(cfg["aircraft"]["actuators"]["flap"]["max_deg"])
        st = {"pos": pos, "R": R, "nav": nav, "frame": frame, "detection": det, "debug": debug, "info": info,
              "estimate": s.guid.tracker.est, "flap": float(flap)}
        ex.record(st)
        st["hud"] = [f"t {nav.t:5.1f} s   alt {-pos[2]:6.1f} m   airspeed {nav.airspeed:4.1f} m/s",
                     f"phase {s.guid.phase}   flap {100 * flap:3.0f} %   runway {'seen' if det is not None else '-'}"]
        if not window:
            ex.view = 0 if pos[2] < -60 else 1  # chase while high, tower view for the landing
            rec.write(ex.compose(st))
            return None
        while True:
            st["hud"] = st["hud"][:2] + (["PAUSED (space)"] if state["paused"] else [])
            img = ex.compose(st)
            if not state["paused"]:
                busy = time.perf_counter() - state["last"]
                state["interval"] = max(busy, 0.9 * state["interval"] + 0.1 * busy)
                if state["interval"] - busy > 0:
                    time.sleep(state["interval"] - busy)
                state["last"] = time.perf_counter()
            cv2.imshow(WINDOW, img)
            if not state["paused"]:
                rec.write(img)
            key = cv2.waitKey(1 if not state["paused"] else 50) & 0xFF
            if key in (ord("q"), 27):
                return False
            if key == ord(" "):
                state["paused"] = not state["paused"]
            if key == ord("c") and s.mode != "vision":
                state["cv"] = not state["cv"]
            ex.handle_key(key)
            if not state["paused"]:
                return None

    sim.frame_callback = on_frame
    sim.run()
    if window:
        print("touchdown - press any key in the window to close")
        cv2.waitKey(0)


def run_free(cfg, rec):
    r = cfg["runway"]
    rwy = Runway(r["center_ne_m"], r["heading_deg"], r["length_m"], r["width_m"])
    seed = cfg["environment"].get("seed", 0)
    print("generating world ...")
    world = World(rwy, seed=seed)
    cam = OmniCamera(cfg["camera"])
    ex = Explorer(world, AircraftParams.from_config(cfg["aircraft"]), cam, rwy, seed,
                  airspeed=cfg["guidance"]["airspeed_mps"])
    attach_mouse(ex)
    detector = RunwayDetector(cam)
    cv_on = True

    # start on a 3 deg approach 600 m out, lined up with the runway
    pos = np.array([*(rwy.threshold(1) - 600 * rwy.axis), -35.0])
    euler = np.array([0.0, np.radians(-3), rwy.heading])
    t = 0.0
    while True:
        R = rot_from_euler(*euler)
        nav = NavEstimate(t, pos.copy(), R[:, 0] * 12.0, euler.copy(), np.zeros(3), 12.0)
        frame = ex.fish_renderer.render(*cam.pose(pos, R))
        det = detector.detect(frame, nav) if cv_on else None
        st = {"pos": pos, "R": R, "nav": nav, "frame": frame, "detection": det,
              "debug": detector.last_debug if cv_on else None, "info": {}, "estimate": None, "flap": 0.0}
        ex.record(st)
        st["hud"] = [f"alt {-pos[2]:6.1f} m   N {pos[0]:7.1f}  E {pos[1]:7.1f}   hdg {np.degrees(euler[2]) % 360:5.1f}"
                     f"  pitch {np.degrees(euler[1]):5.1f}  roll {np.degrees(euler[0]):5.1f}",
                     "w/s move  a/d yaw  r/f up/down  i/k pitch  j/l roll  (shift = 5x)"]
        img = ex.compose(st)
        cv2.imshow(WINDOW, img)
        rec.write(img)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        ex.handle_key(key)
        if key == ord("c"):
            cv_on = not cv_on
        k = chr(key) if key < 128 else ""
        step = 50.0 if k.isupper() else 10.0
        k = k.lower()
        t += 1.0
        if k == "w":
            pos += step * R[:, 0]
        elif k == "s":
            pos -= step * R[:, 0]
        elif k == "r":
            pos[2] -= step
        elif k == "f":
            pos[2] = min(pos[2] + step, -1.0)
        elif k == "a":
            euler[2] -= np.radians(10)
        elif k == "d":
            euler[2] += np.radians(10)
        elif k == "i":
            euler[1] = max(euler[1] - np.radians(5), np.radians(-85))
        elif k == "k":
            euler[1] = min(euler[1] + np.radians(5), np.radians(85))
        elif k == "j":
            euler[0] -= np.radians(10)
        elif k == "l":
            euler[0] += np.radians(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default_glider.yaml")
    ap.add_argument("--free", action="store_true", help="fly around yourself instead of watching the autopilot")
    ap.add_argument("--oracle", action="store_true", help="live mode: guidance uses truth-based perception")
    ap.add_argument("--record", default=None, help="save the dashboard to this .mp4")
    ap.add_argument("--no-window", action="store_true", help="live mode: just record (needs --record)")
    ap.add_argument("--fast", action="store_true",
                    help="smoother: half-res outside view, truth-based perception, CV panels off (press c)")
    args = ap.parse_args()

    overrides = {"perception.mode": "oracle"} if (args.oracle or args.fast) else {}
    if args.fast:
        overrides["camera.rate_hz"] = 10.0  # more frames per simulated second -> ~real-time motion
    cfg = load_config(args.config, overrides)
    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    rec = Recorder(args.record, cfg["camera"]["rate_hz"] if not args.free else 10)
    try:
        if args.free:
            run_free(cfg, rec)
        else:
            run_live(cfg, rec, window=not args.no_window, render_scale=0.5 if args.fast else 1.0,
                     cv_preview=not args.fast)
    finally:
        rec.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
