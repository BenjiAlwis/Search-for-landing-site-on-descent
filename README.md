# Drogue-glider landing simulator

Closed-loop simulation of a rocket's top-stage glider that uses a belly-mounted
360° fisheye ("bulge eye") camera to find a runway and fly itself onto it.

```
 physics (6-DOF, RK4 @200 Hz) ──► sensors (noisy nav) ──► guidance ──► autopilot ──► actuators ─┐
        ▲                              │                    ▲                                  │
        │                              ▼                    │                                  │
        │                  camera render (5 Hz) ──► runway detector                            │
        └──────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/run_sim.py                  # full vision-in-the-loop run (~minutes)
.venv/bin/python scripts/run_sim.py --oracle --no-video   # fast run (~10 s), no rendering
.venv/bin/python scripts/generate_dataset.py --n 500 # labelled fisheye images for CV training
.venv/bin/python -m pytest -q tests
```

### Exploring the environment

```bash
.venv/bin/python scripts/explore.py          # watch the autopilot fly, live
.venv/bin/python scripts/explore.py --free   # fly around the world yourself (no physics)
```

The window is a grid of nine panels. **Click a tile (or press 1-9) to show it large.**

| # | Panel | Shows |
|---|---|---|
| 1 | outside | chase / tower / side / overhead camera with the glider (`v` cycles, `+`/`-` zoom) |
| 2 | eye | onboard fisheye: true runway (green) vs detected (red) |
| 3 | birds-eye | fisheye re-projected onto the ground plane (detector step 1) |
| 4 | saturation | HSV saturation heat-map and the S < 40 threshold (step 2) |
| 5 | segmentation | asphalt / paint classes and the cleaned mask (steps 2-3) |
| 6 | candidates | every shape's fitted rectangle with its accept / reject reason (step 4) |
| 7 | map | flight path, runway, energy orbit, FAF, aim point, runway estimate |
| 8 | glide | altitude vs distance-to-go inside the achievable glide cone, with the live equations |
| 9 | timeline | required vs actual glide angle, flap, detections, runway-estimate error vs truth |

Other keys: `q` quits. Live mode: `space` pauses. `--fast` runs smoother but turns
the CV panels off (`c` turns them back on). Free mode: `w/s` move, `a/d` yaw,
`r/f` up/down, `i/k` pitch, `j/l` roll, and Shift moves 5× faster.
Add `--record out/explore.mp4` to save a video.

Outputs go to `out/`: `flight.png` (track, altitude, attitude, surfaces),
`camera.mp4` (onboard view with detection overlay), `touchdown.json`.

Override any config value from the command line:
`--set environment.wind_ned_mps=[0,-5,0] --set environment.turbulence=moderate`

## Plugging in your glider

Everything aircraft-specific is in [config/default_glider.yaml](config/default_glider.yaml)
(the current numbers are a placeholder 2 m / 2 kg glider):

| Block | What to provide |
|---|---|
| `aircraft` | mass, inertia, wing area/span/chord, Oswald e |
| `aircraft.aero` | stability & control derivatives (from AVL/XFLR5/OpenVSP, wind tunnel or flight test) |
| `aircraft.mixer` | `conventional`, `elevon` (flying wing) or `vtail` |
| `aircraft.actuators` | servo limits, rate, lag |
| `camera` | lens model + FOV + resolution + mount angles; `kannala_brandt` takes `cv2.fisheye` calibration coefficients directly |
| `initial_state` | the state at separation from the rocket |

The autopilot gains are **derived from the aero derivatives** (successive loop
closure), so they re-tune automatically for new specs.

## What's modelled

- **Flight dynamics**: full 6-DOF rigid body, quaternion attitude, RK4. Stability-derivative
  aero with parabolic drag polar, post-stall flat-plate blending, flap increments and
  ground effect. Flat Earth, NED frame.
- **Atmosphere**: US Standard Atmosphere 1976 to 47 km, steady wind, Dryden turbulence (MIL-F-8785C).
- **Actuators**: mixing, position/rate limits, first-order servo lag.
- **Sensors**: navigation estimate with noise and drifting biases. Guidance never sees truth.
- **Camera**: equidistant / equisolid / Kannala-Brandt fisheye or equirectangular 360,
  arbitrary mount, ray-cast onto mip-mapped procedural farmland plus a runway with real
  markings (piano keys, numbers, centerline), haze and sensor noise.
- **Perception** (`glider_sim/vision.py`): baseline classical detector. It projects the fisheye
  image to a bird's-eye view using the nav attitude/altitude, segments asphalt and paint,
  and fits a rectangle. `oracle` mode replaces it with truth plus noise for fast iteration.
- **Guidance** (`glider_sim/guidance.py`): runway tracking, into-wind direction choice,
  energy-management orbit tangent to the final approach, straight-in final, flare and
  rudder decrab. Glide path is held with **flaps** and airspeed with **elevator**, as on a real glider.

## Swapping in your own CV model

Implement an object with `detect(image, nav) -> RunwayDetection | None`
(see `RunwayDetector` in `glider_sim/vision.py`) and assign it to `sim.detector`.
`generate_dataset.py` writes images with per-image JSON labels (runway corners and
center in pixels, camera pose, runway geometry) for training.

## Known simplifications

Flat Earth and flat ground (no terrain relief or 3D objects), no airframe occlusion
in the camera view, no lighting/shadows/sun glare, and no ground roll after touchdown
(the sim stops at touchdown).
