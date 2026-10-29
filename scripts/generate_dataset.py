"""Generate labelled fisheye images for training a runway detector.

Samples random glider poses around the runway (altitude, offset, attitude),
renders the camera view and writes per-image JSON labels with the runway
corners / centre in pixel coordinates plus the camera pose. Each world seed
gives different terrain, and runway heading/size are randomised too.

  python scripts/generate_dataset.py --n 500 --out data/runway_fisheye
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glider_sim.camera import OmniCamera, Renderer  # noqa: E402
from glider_sim.rotations import rot_from_euler  # noqa: E402
from glider_sim.sim import load_config  # noqa: E402
from glider_sim.vision import runway_label  # noqa: E402
from glider_sim.world import Runway, World  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_glider.yaml")
    ap.add_argument("--out", default="data/runway_fisheye")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--worlds", type=int, default=4, help="number of distinct terrains/runways")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cam = OmniCamera(cfg["camera"])

    per_world = int(np.ceil(args.n / args.worlds))
    idx = 0
    for w in range(args.worlds):
        rwy = Runway([0, 0], rng.uniform(0, 180), rng.uniform(200, 800), rng.uniform(15, 45))
        world = World(rwy, seed=args.seed * 1000 + w)
        ren = Renderer(cam, world, seed=w)
        for _ in range(per_world):
            if idx >= args.n:
                break
            h = float(np.exp(rng.uniform(np.log(3), np.log(1200))))
            rng_xy = rng.uniform(0, max(150, 6 * h))
            brg = rng.uniform(-np.pi, np.pi)
            pos = np.array([rng_xy * np.cos(brg), rng_xy * np.sin(brg), -h])
            eul = np.array([rng.normal(0, 0.35), rng.normal(-0.1, 0.2), rng.uniform(-np.pi, np.pi)])
            cam_pos, R = cam.pose(pos, rot_from_euler(*eul))
            img = ren.render(cam_pos, R)
            name = f"{idx:06d}"
            cv2.imwrite(str(out / "images" / f"{name}.png"), img)
            label = runway_label(cam, rwy, cam_pos, R)
            label.update({"image": f"images/{name}.png", "altitude_m": h, "position_ned_m": pos.tolist(),
                          "euler_rad": eul.tolist(), "runway": {"heading_deg": float(np.degrees(rwy.heading)),
                                                                "length_m": rwy.length, "width_m": rwy.width},
                          "camera": {k: cfg["camera"][k] for k in ("projection", "model", "fov_deg",
                                                                   "width_px", "height_px", "mount_rpy_deg")}})
            (out / "images" / f"{name}.json").write_text(json.dumps(label))
            idx += 1
        print(f"world {w + 1}/{args.worlds}: {idx} images")


if __name__ == "__main__":
    main()
