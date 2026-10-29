"""Run one closed-loop descent-and-landing simulation.

Examples
  python scripts/run_sim.py                              # vision mode, plots + camera video
  python scripts/run_sim.py --oracle                     # fast: skip rendering, truth+noise detections
  python scripts/run_sim.py --set environment.wind_ned_mps=[0,-5,0] --set environment.turbulence=moderate
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glider_sim.plotting import plot_flight  # noqa: E402
from glider_sim.sim import Simulation, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_glider.yaml")
    ap.add_argument("--out", default="out")
    ap.add_argument("--oracle", action="store_true", help="truth-based perception, no rendering")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. guidance.airspeed_mps=13")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    if args.oracle:
        overrides["perception.mode"] = "oracle"
        if args.no_video:
            overrides["camera.enabled"] = False
    cfg = load_config(args.config, overrides)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video = None if args.no_video else str(out / "camera.mp4")
    sim = Simulation(cfg, video_path=video)
    res = sim.run()
    plot_flight(res, sim.runway, out / "flight.png")
    (out / "touchdown.json").write_text(json.dumps({"touchdown": res.touchdown, "gains": res.gains}, indent=2))
    print(f"wrote {out / 'flight.png'}" + (f", {video}" if video else ""))


if __name__ == "__main__":
    main()
