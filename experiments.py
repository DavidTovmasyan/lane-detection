"""Run the full experiment grid and produce CSVs + figures.

Three experiments are executed:

  E1) Method comparison: every (detector, controller) pair on every
      track. Outputs ``results/summary.csv`` and the headline figures.

  E2) Noise robustness: PID controller, all three detectors, on the
      oval track, with Gaussian camera noise sigma in {0, 0.02, 0.05,
      0.10, 0.20}. Outputs ``results/noise.csv``.

  E3) Per-frame trajectory log on the snake track for the Hough and
      Polyfit detectors (PID), used for the offset-history plot.

All figures are written to ``presentation/figures/`` so the LaTeX
sources pick them up directly.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from evaluate import run_one


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "presentation" / "results"
FIGURES = ROOT / "presentation" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


TRACKS = ["oval", "stadium", "snake", "grand_prix", "mountain"]
DETECTORS = ["centroid", "hough", "polyfit"]
CONTROLLERS = ["pid", "stanley"]


# ──────────────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "centroid": "#888888",
    "hough":    "#d6604d",
    "polyfit":  "#2c7fb8",
    "stanley":  "#2ca25f",
}


# ──────────────────────────────────────────────────────────────────────
# E1: method-comparison grid
# ──────────────────────────────────────────────────────────────────────

def experiment_method_comparison():
    rows = []
    print("\n[E1] method comparison")
    for track in TRACKS:
        for det in DETECTORS:
            for ctl in CONTROLLERS:
                s = run_one(det, ctl, track, laps=1, max_duration_s=120.0)
                rows.append(s.as_dict())
                ok = "OK " if s.completed_lap else "DNF"
                print(f"  {track:12s} {det:9s} {ctl:8s} {ok}"
                      f"  t={s.duration_s:6.1f}s  rms={s.rms_off_px:6.2f}px"
                      f"  iou={s.mean_iou:.3f}  ms/det={s.mean_detect_ms:.2f}")
    _write_csv(RESULTS / "summary.csv", rows)
    _plot_rms_by_track(rows)
    _plot_iou_by_track(rows)
    _plot_method_summary_table(rows)
    return rows


# ──────────────────────────────────────────────────────────────────────
# E2: noise robustness
# ──────────────────────────────────────────────────────────────────────

def experiment_noise():
    rows = []
    sigmas = [0.0, 0.02, 0.05, 0.10, 0.20]
    print("\n[E2] noise robustness")
    for det in DETECTORS:
        for sigma in sigmas:
            s = run_one(det, "pid", "oval",
                        laps=1, max_duration_s=60.0, noise=sigma)
            d = s.as_dict()
            d["sigma"] = sigma
            rows.append(d)
            print(f"  oval  {det:9s} sigma={sigma:.2f} "
                  f"completed={s.completed_lap}  rms={s.rms_off_px:6.2f}"
                  f"  off={s.off_track_rate*100:5.1f}%")
    _write_csv(RESULTS / "noise.csv", rows)
    _plot_noise(rows, sigmas)


# ──────────────────────────────────────────────────────────────────────
# E3: trajectory snapshot for offset history figure
# ──────────────────────────────────────────────────────────────────────

def experiment_traces():
    print("\n[E3] per-frame trajectories on snake (PID)")
    for det in DETECTORS:
        out = RESULTS / f"trace_snake_{det}_pid.csv"
        s = run_one(det, "pid", "snake",
                    laps=1, max_duration_s=60.0,
                    log_path=str(out))
        print(f"  snake {det:9s} -> {out.name}  "
              f"laps={s.laps_done} rms={s.rms_off_px:.2f}")
    _plot_traces()


# ──────────────────────────────────────────────────────────────────────
# CSV writer
# ──────────────────────────────────────────────────────────────────────

def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────

def _plot_rms_by_track(rows):
    # PID rows only (Stanley is an oracle reference; we plot it separately)
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(TRACKS))
    bar_w = 0.25
    for i, det in enumerate(DETECTORS):
        vals = []
        for tr in TRACKS:
            r = next((r for r in rows
                      if r["detector"] == det and r["controller"] == "pid"
                      and r["track"] == tr), None)
            vals.append(r["rms_off_px"] if r and r["completed_lap"] else np.nan)
        ax.bar(x + (i - 1) * bar_w, vals, bar_w,
               label=det, color=COLORS[det], edgecolor="black", linewidth=0.4)
    # Stanley oracle: averaged across detectors (they're identical)
    oracle = []
    for tr in TRACKS:
        rr = [r for r in rows if r["track"] == tr and r["controller"] == "stanley"]
        if rr:
            oracle.append(np.mean([r["rms_off_px"] for r in rr]))
        else:
            oracle.append(np.nan)
    ax.plot(x, oracle, marker="o", linestyle="--",
            color=COLORS["stanley"], label="Stanley (oracle)",
            linewidth=1.5, markersize=5)

    ax.set_xticks(x)
    ax.set_xticklabels([cfg.TRACKS[t]["description"] for t in TRACKS],
                       rotation=15, ha="right")
    ax.set_ylabel("RMS lateral error (px)")
    ax.set_title("Lateral RMS error per track (PID controller)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    # DNF markers
    for i, det in enumerate(DETECTORS):
        for j, tr in enumerate(TRACKS):
            r = next((r for r in rows
                      if r["detector"] == det and r["controller"] == "pid"
                      and r["track"] == tr), None)
            if r and not r["completed_lap"]:
                ax.text(j + (i - 1) * bar_w, 5, "DNF",
                        ha="center", va="bottom",
                        color="black", fontsize=7, rotation=90)
    plt.tight_layout()
    fig.savefig(FIGURES / "rms_by_track.pdf")
    fig.savefig(FIGURES / "rms_by_track.png", dpi=130)
    plt.close(fig)


def _plot_iou_by_track(rows):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(TRACKS))
    bar_w = 0.32
    for i, det in enumerate(["hough", "polyfit"]):
        vals = []
        for tr in TRACKS:
            r = next((r for r in rows
                      if r["detector"] == det and r["controller"] == "pid"
                      and r["track"] == tr), None)
            vals.append(r["mean_iou"] if r else 0.0)
        ax.bar(x + (i - 0.5) * bar_w, vals, bar_w,
               label=det, color=COLORS[det], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([cfg.TRACKS[t]["description"] for t in TRACKS],
                       rotation=15, ha="right")
    ax.set_ylabel("Mean lane IoU")
    ax.set_title("Predicted-vs-true lane IoU (PID controller)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "iou_by_track.pdf")
    fig.savefig(FIGURES / "iou_by_track.png", dpi=130)
    plt.close(fig)


def _plot_method_summary_table(rows):
    """Produce a numerical summary written to a .tex fragment."""
    out = ["\\begin{tabular}{l l r r r r r r}",
           "\\toprule",
           "Track & Detector & Lap & RMS px & Mean px & Max px & IoU & ms/det \\\\",
           "\\midrule"]
    for tr in TRACKS:
        for det in DETECTORS:
            r = next((r for r in rows
                      if r["detector"] == det and r["controller"] == "pid"
                      and r["track"] == tr), None)
            if not r:
                continue
            ok = "\\checkmark" if r["completed_lap"] else "DNF"
            out.append(
                f"{cfg.TRACKS[tr]['description']} & {det} & {ok} "
                f"& {r['rms_off_px']:.1f} & {r['mean_abs_off_px']:.1f} "
                f"& {r['max_abs_off_px']:.1f} & {r['mean_iou']:.2f} "
                f"& {r['mean_detect_ms']:.2f} \\\\"
            )
        out.append("\\midrule")
    # Stanley oracle row
    rs = [r for r in rows if r["controller"] == "stanley" and r["track"] == "oval"]
    if rs:
        r = rs[0]
        out.append(
            f"{cfg.TRACKS['oval']['description']} & Stanley (oracle) & "
            f"\\checkmark & {r['rms_off_px']:.1f} & {r['mean_abs_off_px']:.1f} "
            f"& {r['max_abs_off_px']:.1f} & {r['mean_iou']:.2f} & "
            f"{r['mean_detect_ms']:.2f} \\\\"
        )
    out.append("\\bottomrule")
    out.append("\\end{tabular}")

    (FIGURES / "summary_table.tex").write_text("\n".join(out))


def _plot_noise(rows, sigmas):
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for det in DETECTORS:
        ys = []
        for sigma in sigmas:
            r = next((r for r in rows
                      if r["detector"] == det and r["sigma"] == sigma), None)
            ys.append(r["rms_off_px"] if r and r["completed_lap"] else np.nan)
        ax.plot(sigmas, ys, marker="o",
                color=COLORS[det], label=det, linewidth=1.5)
    ax.set_xlabel("Gaussian noise $\\sigma$ (fraction of intensity)")
    ax.set_ylabel("RMS lateral error (px)")
    ax.set_title("Noise robustness on the oval (PID controller)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "noise.pdf")
    fig.savefig(FIGURES / "noise.png", dpi=130)
    plt.close(fig)


def _plot_traces():
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True)

    for det in DETECTORS:
        path = RESULTS / f"trace_snake_{det}_pid.csv"
        if not path.exists():
            continue
        rows = list(_read_csv(path))
        t = [r["t"] for r in rows]
        true_off = [r["true_offset"] for r in rows]
        det_off = [r["det_offset"] for r in rows]
        axs[0].plot(t, true_off, color=COLORS[det], label=det, linewidth=1.0)
        axs[1].plot(t, det_off, color=COLORS[det], label=det, linewidth=1.0, alpha=0.85)

    for ax in axs:
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

    axs[0].set_ylabel("True lateral err (px)")
    axs[0].set_title("Snake track: true vs detected lateral error (PID controller)")
    axs[0].legend(ncol=3, frameon=False, loc="upper right")
    axs[1].set_ylabel("Detector offset (px)")
    axs[1].set_xlabel("time (s)")
    plt.tight_layout()
    fig.savefig(FIGURES / "snake_traces.pdf")
    fig.savefig(FIGURES / "snake_traces.png", dpi=130)
    plt.close(fig)


def _read_csv(path):
    with open(path) as f:
        for row in csv.DictReader(f):
            yield {k: (float(v) if v not in ("", "True", "False")
                       else v == "True") for k, v in row.items()}


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    e1 = experiment_method_comparison()
    experiment_noise()
    experiment_traces()
    print("\nDone. Figures written to", FIGURES)
