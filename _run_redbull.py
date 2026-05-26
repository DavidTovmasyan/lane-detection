"""One-off: run the missing redbull_ring experiments and regenerate figures.

Loads the existing summary.csv, runs E1 for redbull_ring only, appends rows,
rewrites summary.csv, regenerates the per-track bar charts and the summary
LaTeX table, then runs E3 traces for redbull_ring and produces the trace
figure. Idempotent: existing redbull rows are dropped before re-running.
"""
from __future__ import annotations

import csv
from pathlib import Path

from evaluate import run_one
from experiments import (
    DETECTORS, CONTROLLERS, FIGURES, RESULTS,
    _plot_iou_by_track, _plot_method_summary_table,
    _plot_rms_by_track, _plot_traces, _write_csv,
)

TRACK = "redbull_ring"
SUMMARY = RESULTS / "summary.csv"


def _typed(row: dict) -> dict:
    """Coerce a csv.DictReader row to the types the plot helpers expect."""
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
        elif v in ("True", "False"):
            out[k] = (v == "True")
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def load_summary() -> list[dict]:
    if not SUMMARY.exists():
        return []
    with open(SUMMARY) as f:
        return [_typed(r) for r in csv.DictReader(f)]


def main():
    existing = [r for r in load_summary() if r.get("track") != TRACK]
    print(f"[setup] kept {len(existing)} rows from non-{TRACK} tracks")

    print(f"\n[E1] method comparison on {TRACK}")
    new_rows = []
    for det in DETECTORS:
        for ctl in CONTROLLERS:
            s = run_one(det, ctl, TRACK, laps=1, max_duration_s=180.0)
            new_rows.append(s.as_dict())
            ok = "OK " if s.completed_lap else "DNF"
            print(f"  {TRACK:13s} {det:9s} {ctl:8s} {ok}"
                  f"  t={s.duration_s:6.1f}s  rms={s.rms_off_px:6.2f}px"
                  f"  iou={s.mean_iou:.3f}  ms/det={s.mean_detect_ms:.2f}")

    combined = existing + new_rows
    _write_csv(SUMMARY, combined)
    print(f"[E1] wrote {len(combined)} rows -> {SUMMARY}")

    _plot_rms_by_track(combined)
    _plot_iou_by_track(combined)
    _plot_method_summary_table(combined)
    print(f"[E1] regenerated rms/iou bar charts and summary_table.tex")

    print(f"\n[E3] per-frame trajectory on {TRACK} (PID)")
    for det in DETECTORS:
        out = RESULTS / f"trace_{TRACK}_{det}_pid.csv"
        s = run_one(det, "pid", TRACK, laps=1, max_duration_s=120.0,
                    log_path=str(out))
        print(f"  {TRACK:13s} {det:9s} -> {out.name}  "
              f"laps={s.laps_done} rms={s.rms_off_px:.2f}")
    _plot_traces(TRACK)
    print(f"[E3] wrote {FIGURES / f'{TRACK}_traces.pdf'}")


if __name__ == "__main__":
    main()
