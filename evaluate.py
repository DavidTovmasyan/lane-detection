"""Headless evaluation harness.

Runs the simulator without a display window and produces quantitative
metrics for a given (detector, controller, track) combination.

Metrics logged per frame
------------------------
    t                 elapsed time (s)
    x, y              vehicle position
    heading           vehicle heading (rad)
    speed             vehicle speed (px/s)
    true_offset       signed lateral offset from track centreline (px)
    true_heading_err  heading error vs track tangent (rad)
    det_offset        detector lateral offset (camera px)
    det_lookahead     detector look-ahead offset (camera px)
    det_confidence    detector confidence in [0,1]
    steering          commanded steering angle (rad)
    lane_iou          IoU between predicted lane mask and ground-truth
                      road-polygon mask (in camera image)
    off_track         1 if |true_offset| > 0.5 * road_width else 0
    detect_ms         detector inference time (ms)

Aggregated metrics
------------------
    mean_abs_off_px       Mean |true lateral error| (px)
    rms_off_px            RMS lateral error
    max_abs_off_px        Maximum lateral error
    mean_iou              Mean lane IoU
    mean_conf             Mean detection confidence
    off_track_rate        Fraction of frames the car was off the road
    completed_lap         True if the car closed a loop without leaving
                          the road for >1 s
    lap_time_s            Time to first lap completion (None if dnf)
    mean_detect_ms        Mean detector inference time
    fps_equiv             Equivalent FPS of the detector

Usage
-----
    python evaluate.py --detector polyfit --controller pid --track oval \\
        --laps 1 --noise 0.0 --out results/oval_polyfit_pid.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

import config as cfg
from camera import FPVCamera
from controller import PIDController, StanleyController
from lane_methods import make_detector
from track import Track
from vehicle import Vehicle


# ──────────────────────────────────────────────────────────────────────
# Ground-truth lane mask (re-projects the simulator's road polygon).
# ──────────────────────────────────────────────────────────────────────

def _gt_lane_mask(camera: FPVCamera, track: Track,
                  car_x: float, car_y: float, heading: float) -> np.ndarray:
    h, w = camera.h, camera.w
    left, right, _ = track.get_road_ahead(car_x, car_y, camera.look_ahead)
    if len(left) < 2 or len(right) < 2:
        return np.zeros((h, w), dtype=np.uint8)

    left_uv, lv = camera._project(left, car_x, car_y, heading)
    right_uv, rv = camera._project(right, car_x, car_y, heading)
    left_uv = left_uv[lv].astype(np.int32)
    right_uv = right_uv[rv].astype(np.int32)
    if len(left_uv) < 2 or len(right_uv) < 2:
        return np.zeros((h, w), dtype=np.uint8)
    poly = np.vstack([left_uv, right_uv[::-1]])
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 255)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    ab = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(ab / union) if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# Frame noise (used for the robustness sweep).
# ──────────────────────────────────────────────────────────────────────

def _add_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma * 255.0, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
# Aggregate result container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RunSummary:
    detector: str
    controller: str
    track: str
    noise: float
    laps_target: int
    laps_done: int
    completed_lap: bool
    lap_time_s: Optional[float]
    duration_s: float
    n_frames: int
    mean_abs_off_px: float
    rms_off_px: float
    max_abs_off_px: float
    mean_iou: float
    mean_conf: float
    off_track_rate: float
    mean_detect_ms: float
    fps_equiv: float

    def as_dict(self):
        return self.__dict__.copy()


# ──────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────

def run_one(detector_name: str,
            controller_name: str,
            track_key: str,
            *,
            laps: int = 1,
            dt: float = 1 / 60,
            max_duration_s: float = 240.0,
            noise: float = 0.0,
            log_path: Optional[str] = None,
            verbose: bool = False) -> RunSummary:
    tdef = cfg.TRACKS[track_key]
    track = Track(tdef["waypoints"], tdef["road_width"])
    vehicle = Vehicle(track)
    base_speed = tdef["speed"]
    vehicle.speed = base_speed
    camera = FPVCamera()
    detector = make_detector(detector_name)
    pid = PIDController()
    stanley = StanleyController()

    # Lap detection: start near centerline index 0, track when we leave
    # a neighbourhood of the start and then return to it.
    start = track.centerline[0]
    left_start = False
    laps_done = 0
    lap_time = None
    detached_t = None  # time when we first went off-road

    rows = []
    sum_abs_off = sum_off2 = max_abs_off = 0.0
    sum_iou = sum_conf = 0.0
    off_track_frames = 0
    detect_ms_total = 0.0
    n = 0
    completed_lap = False

    t = 0.0
    while t < max_duration_s and laps_done < laps:
        img = camera.render(vehicle.x, vehicle.y, vehicle.heading, track)
        img = _add_noise(img, noise) if noise > 0 else img

        t0 = time.perf_counter()
        det = detector.detect(img)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        detect_ms_total += detect_ms

        if controller_name == "stanley":
            steer = stanley.compute(track, vehicle.x, vehicle.y,
                                    vehicle.heading, vehicle.speed, dt)
        else:
            err_px = (det.lookahead_offset
                      if detector_name == "polyfit"
                      else det.center_offset)
            err_norm = err_px / (cfg.CAMERA_IMG_W / 2)
            steer = pid.compute(err_norm, dt)

        vehicle.update(dt, steer)

        # Ground-truth metrics
        idx, true_off = track.get_nearest_index(vehicle.x, vehicle.y)
        track_h = track.get_heading_at(idx)
        heading_err = math.atan2(math.sin(track_h - vehicle.heading),
                                  math.cos(track_h - vehicle.heading))

        # Lane IoU
        gt_mask = _gt_lane_mask(camera, track, vehicle.x, vehicle.y,
                                vehicle.heading)
        iou = _iou(det.pred_mask, gt_mask)

        off = abs(true_off) > 0.5 * track.road_width
        if off:
            off_track_frames += 1
            if detached_t is None:
                detached_t = t
        else:
            detached_t = None

        sum_abs_off += abs(true_off)
        sum_off2 += true_off * true_off
        max_abs_off = max(max_abs_off, abs(true_off))
        sum_iou += iou
        sum_conf += det.confidence
        n += 1

        # Lap detection: must leave a 60-px radius of start, then come back.
        d_start = math.hypot(vehicle.x - start[0], vehicle.y - start[1])
        if d_start > 100:
            left_start = True
        if left_start and d_start < 40:
            laps_done += 1
            left_start = False
            if lap_time is None:
                lap_time = t
                completed_lap = True

        # Hard fail: off track for >1.5 s
        if detached_t is not None and (t - detached_t) > 1.5:
            if verbose:
                print(f"  [{detector_name}/{controller_name}/{track_key}] off-track @ t={t:.1f}s")
            break

        if log_path is not None:
            rows.append({
                "t": t,
                "x": vehicle.x,
                "y": vehicle.y,
                "heading": vehicle.heading,
                "speed": vehicle.speed,
                "true_offset": true_off,
                "true_heading_err": heading_err,
                "det_offset": det.center_offset,
                "det_lookahead": det.lookahead_offset,
                "det_confidence": det.confidence,
                "steering": vehicle.steering,
                "lane_iou": iou,
                "off_track": int(off),
                "detect_ms": detect_ms,
            })

        t += dt

    if log_path is not None and rows:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n_div = max(n, 1)
    return RunSummary(
        detector=detector_name,
        controller=controller_name,
        track=track_key,
        noise=noise,
        laps_target=laps,
        laps_done=laps_done,
        completed_lap=completed_lap,
        lap_time_s=lap_time,
        duration_s=t,
        n_frames=n,
        mean_abs_off_px=sum_abs_off / n_div,
        rms_off_px=math.sqrt(sum_off2 / n_div),
        max_abs_off_px=max_abs_off,
        mean_iou=sum_iou / n_div,
        mean_conf=sum_conf / n_div,
        off_track_rate=off_track_frames / n_div,
        mean_detect_ms=detect_ms_total / n_div,
        fps_equiv=1000.0 / max(detect_ms_total / n_div, 1e-3),
    )


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Headless lane-detection evaluator")
    p.add_argument("--detector", default="hough",
                   choices=["centroid", "hough", "polyfit"])
    p.add_argument("--controller", default="pid", choices=["pid", "stanley"])
    p.add_argument("--track", default="oval", choices=list(cfg.TRACKS.keys()))
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian noise sigma in [0,1] applied to the camera frame")
    p.add_argument("--out", type=str, default=None,
                   help="Optional CSV path for the per-frame log")
    p.add_argument("--max-duration", type=float, default=240.0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    summary = run_one(
        args.detector, args.controller, args.track,
        laps=args.laps, max_duration_s=args.max_duration,
        noise=args.noise, log_path=args.out, verbose=args.verbose,
    )

    print(f"\n=== {args.detector}/{args.controller}/{args.track}"
          f"  noise={args.noise} ===")
    for k, v in summary.as_dict().items():
        if isinstance(v, float):
            print(f"  {k:<22s} {v:.4f}")
        else:
            print(f"  {k:<22s} {v}")


if __name__ == "__main__":
    main()
