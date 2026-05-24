"""Generate a more compact summary table (PID rows only)."""
import csv
from pathlib import Path

import config as cfg

R = Path(__file__).resolve().parent / "results" / "summary.csv"
OUT = Path(__file__).resolve().parent / "figures" / "summary_table.tex"


def f(v):
    try:
        return float(v)
    except Exception:
        return v


rows = list(csv.DictReader(open(R)))
by_key = {(r["track"], r["detector"], r["controller"]): r for r in rows}

TRACKS = ["oval", "stadium", "snake", "grand_prix", "mountain"]

out = [
    r"\begin{tabular}{l c c c c c c}",
    r"\toprule",
    r" & \multicolumn{2}{c}{\textsc{hough}} & \multicolumn{2}{c}{\textsc{polyfit}}"
    r" & \textsc{centroid} & Stanley \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"Track & RMS & IoU & RMS & IoU & RMS & RMS \\",
    r"\midrule",
]
for tr in TRACKS:
    h = by_key.get((tr, "hough", "pid"))
    p = by_key.get((tr, "polyfit", "pid"))
    c = by_key.get((tr, "centroid", "pid"))
    s = by_key.get((tr, "hough", "stanley"))  # any detector; stanley ignores it

    def cell_rms(r):
        if r is None:
            return "--"
        if r["completed_lap"] == "True":
            return f"{float(r['rms_off_px']):.1f}"
        return f"\\textsc{{dnf}}"

    def cell_iou(r):
        if r is None or r["completed_lap"] != "True":
            return "--"
        return f"{float(r['mean_iou']):.2f}"

    out.append(
        f"{cfg.TRACKS[tr]['description']} & "
        f"{cell_rms(h)} & {cell_iou(h)} & "
        f"{cell_rms(p)} & {cell_iou(p)} & "
        f"{cell_rms(c)} & {cell_rms(s)} \\\\"
    )
out += [
    r"\midrule",
    r"\multicolumn{1}{l}{\textit{mean} (px)} &",
]


def mean(det, ctl, field="rms_off_px"):
    vals = []
    for tr in TRACKS:
        r = by_key.get((tr, det, ctl))
        if r and r["completed_lap"] == "True":
            vals.append(float(r[field]))
    return sum(vals) / len(vals) if vals else float("nan")


def mean_cell(v):
    if v != v:  # NaN
        return "--"
    return f"{v:.1f}"


def mean_cell_iou(v):
    if v != v:
        return "--"
    return f"{v:.2f}"


out.append(
    f" {mean_cell(mean('hough','pid'))} & {mean_cell_iou(mean('hough','pid','mean_iou'))} & "
    f"{mean_cell(mean('polyfit','pid'))} & {mean_cell_iou(mean('polyfit','pid','mean_iou'))} & "
    f"{mean_cell(mean('centroid','pid'))} & {mean_cell(mean('hough','stanley'))} \\\\"
)
out += [r"\bottomrule", r"\end{tabular}"]
OUT.write_text("\n".join(out))
print("Wrote", OUT)
