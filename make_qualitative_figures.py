"""Render qualitative figures used in the report and slides.

  * ``annotations.png`` -- side-by-side annotated camera frames for the
    three detectors at the same frame on the snake track.
  * ``trajectories.png`` -- top-down map of the snake track with the
    actual driven trajectories overlaid for the three detectors (PID).
  * ``polyfit_pipeline.png`` -- multi-panel visualisation of the
    proposed pipeline: raw frame, lane-pixel mask, bird's-eye view,
    sliding-window fit, lane reprojection.
"""

from __future__ import annotations

import os
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

import config as cfg
from camera import FPVCamera
from controller import PIDController
from lane_methods import (
    make_detector,
    PolyFitDetector,
)
from track import Track
from vehicle import Vehicle


ROOT = Path(__file__).resolve().parent
FIG = ROOT / "presentation" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _drive_trajectory(detector_name: str, track_key: str,
                      n_seconds: float = 60.0, dt: float = 1/60):
    tdef = cfg.TRACKS[track_key]
    track = Track(tdef["waypoints"], tdef["road_width"])
    vehicle = Vehicle(track)
    vehicle.speed = tdef["speed"]
    camera = FPVCamera()
    detector = make_detector(detector_name)
    pid = PIDController()

    pts = [(vehicle.x, vehicle.y)]
    for _ in range(int(n_seconds / dt)):
        img = camera.render(vehicle.x, vehicle.y, vehicle.heading, track)
        det = detector.detect(img)
        err = det.lookahead_offset if detector_name == "polyfit" else det.center_offset
        steer = pid.compute(err / (cfg.CAMERA_IMG_W / 2), dt)
        vehicle.update(dt, steer)
        pts.append((vehicle.x, vehicle.y))
        # Stop early when we're back at the start after leaving it
        dx = vehicle.x - track.centerline[0, 0]
        dy = vehicle.y - track.centerline[0, 1]
        idx, off = track.get_nearest_index(vehicle.x, vehicle.y)
        if abs(off) > 0.6 * track.road_width:
            break
        if len(pts) > 200 and (dx * dx + dy * dy) < 40 ** 2:
            break
    return track, np.array(pts)


def _drive_to_frame(detector_name: str, track_key: str, t_target: float,
                    dt: float = 1/60):
    """Run a detector for ``t_target`` seconds and return the last frame
    and the corresponding vehicle state.
    """
    tdef = cfg.TRACKS[track_key]
    track = Track(tdef["waypoints"], tdef["road_width"])
    vehicle = Vehicle(track)
    vehicle.speed = tdef["speed"]
    camera = FPVCamera()
    detector = make_detector(detector_name)
    pid = PIDController()

    img = None
    for _ in range(int(t_target / dt)):
        img = camera.render(vehicle.x, vehicle.y, vehicle.heading, track)
        det = detector.detect(img)
        err = det.lookahead_offset if detector_name == "polyfit" else det.center_offset
        steer = pid.compute(err / (cfg.CAMERA_IMG_W / 2), dt)
        vehicle.update(dt, steer)
    return img, vehicle, track


# ──────────────────────────────────────────────────────────────────────
# Figure 1 -- annotated camera frames, one per detector
# ──────────────────────────────────────────────────────────────────────

def fig_annotations():
    print("[fig] annotations.png")
    # Drive 6 s on snake with the polyfit detector to land in a curve.
    _, vehicle, track = _drive_to_frame("polyfit", "snake", t_target=6.0)
    cam = FPVCamera()
    frame = cam.render(vehicle.x, vehicle.y, vehicle.heading, track)

    fig, axs = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, name in zip(axs, ["centroid", "hough", "polyfit"]):
        det = make_detector(name)
        # Two passes to give the temporal smoothers a chance.
        det.detect(frame); det.detect(frame)
        d = det.detect(frame)
        rgb = d.annotated
        ax.imshow(rgb)
        ax.set_title(f"{name}  (conf={d.confidence:.2f}, "
                     f"offset={d.center_offset:+.0f}px)")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIG / "annotations.pdf")
    fig.savefig(FIG / "annotations.png", dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Figure 2 -- trajectory overlays on the snake track
# ──────────────────────────────────────────────────────────────────────

def fig_trajectories():
    print("[fig] trajectories.png")
    colors = {"centroid": "#888888", "hough": "#d6604d", "polyfit": "#2c7fb8"}
    fig, ax = plt.subplots(figsize=(5.5, 5.2))

    # Use the first detector to get the track geometry; trajectories
    # come from independent runs of each detector.
    first_track = None
    for det in ["centroid", "hough", "polyfit"]:
        track, traj = _drive_trajectory(det, "snake", n_seconds=40)
        if first_track is None:
            first_track = track
            # Plot left/right boundaries and centerline
            lb = track.left_boundary
            rb = track.right_boundary
            cl = track.centerline
            ax.fill(np.append(lb[:, 0], rb[::-1, 0]),
                    np.append(lb[:, 1], rb[::-1, 1]),
                    color="#dddddd", zorder=1)
            ax.plot(lb[:, 0], lb[:, 1], color="#444444", linewidth=0.7, zorder=2)
            ax.plot(rb[:, 0], rb[:, 1], color="#444444", linewidth=0.7, zorder=2)
            ax.plot(cl[:, 0], cl[:, 1], color="#aaaaaa",
                    linewidth=0.6, linestyle=":", zorder=2)
        ax.plot(traj[:, 0], traj[:, 1], color=colors[det],
                linewidth=1.6, label=det, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[det],
                   marker="o", s=18, zorder=4)
        # mark where the run ended (DNF point)
        if len(traj) < 1000:
            ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[det],
                       marker="x", s=40, zorder=4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Snake track: actual trajectories per detector (PID)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIG / "trajectories.pdf")
    fig.savefig(FIG / "trajectories.png", dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Figure 3 -- pipeline visualisation of the polyfit detector
# ──────────────────────────────────────────────────────────────────────

def fig_polyfit_pipeline():
    print("[fig] polyfit_pipeline.png")
    _, vehicle, track = _drive_to_frame("polyfit", "snake", t_target=6.0)
    cam = FPVCamera()
    frame = cam.render(vehicle.x, vehicle.y, vehicle.heading, track)

    det = PolyFitDetector()
    det.detect(frame); det.detect(frame)
    final = det.detect(frame)

    # Reconstruct intermediate stages
    mask = det._lane_pixels(frame)
    M, Minv = det._warp_matrices(frame.shape[0], frame.shape[1])
    bev = cv2.warpPerspective(mask, M, (det.BEV_W, det.BEV_H),
                              flags=cv2.INTER_NEAREST)

    # Sliding-window overlay on BEV
    win_h = det.BEV_H // det.N_WINDOWS
    nz = bev.nonzero()
    nzy, nzx = nz[0], nz[1]
    hist = np.sum(bev[det.BEV_H // 2:, :], axis=0)
    if hist.max() > 0:
        mid = det.BEV_W // 2
        leftx, rightx = int(np.argmax(hist[:mid])), int(np.argmax(hist[mid:]) + mid)
    else:
        leftx = rightx = det.BEV_W // 2

    bev_vis = cv2.cvtColor(bev, cv2.COLOR_GRAY2RGB)
    for wi in range(det.N_WINDOWS):
        y_lo = det.BEV_H - (wi + 1) * win_h
        y_hi = det.BEV_H - wi * win_h
        cv2.rectangle(bev_vis,
                      (leftx - det.WIN_MARGIN, y_lo),
                      (leftx + det.WIN_MARGIN, y_hi),
                      (0, 255, 100), 1)
        cv2.rectangle(bev_vis,
                      (rightx - det.WIN_MARGIN, y_lo),
                      (rightx + det.WIN_MARGIN, y_hi),
                      (0, 255, 100), 1)
        gl = ((nzy >= y_lo) & (nzy < y_hi)
              & (nzx >= leftx - det.WIN_MARGIN)
              & (nzx < leftx + det.WIN_MARGIN)).nonzero()[0]
        gr = ((nzy >= y_lo) & (nzy < y_hi)
              & (nzx >= rightx - det.WIN_MARGIN)
              & (nzx < rightx + det.WIN_MARGIN)).nonzero()[0]
        if len(gl) > det.MIN_PIX:
            leftx = int(nzx[gl].mean())
        if len(gr) > det.MIN_PIX:
            rightx = int(nzx[gr].mean())

    # Plot
    titles = ["FPV input", "Lane-pixel mask", "Bird's-eye view + sliding windows",
              "Lane reprojected to FPV"]
    images = [frame, mask, bev_vis, final.annotated]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, t, im in zip(axs, titles, images):
        if im.ndim == 2:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIG / "polyfit_pipeline.pdf")
    fig.savefig(FIG / "polyfit_pipeline.png", dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig_annotations()
    fig_trajectories()
    fig_polyfit_pipeline()
    print("Done. Figures in", FIG)
