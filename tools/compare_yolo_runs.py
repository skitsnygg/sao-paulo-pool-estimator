#!/usr/bin/env python3

from pathlib import Path
import csv
import math

ROOTS = [
    Path("runs/segment"),
    Path("runs/segment/runs/segment"),
]

METRIC_MAP = {
    "box50": "metrics/mAP50(B)",
    "box95": "metrics/mAP50-95(B)",
    "mask50": "metrics/mAP50(M)",
    "mask95": "metrics/mAP50-95(M)",
    "p_box": "metrics/precision(B)",
    "r_box": "metrics/recall(B)",
    "p_mask": "metrics/precision(M)",
    "r_mask": "metrics/recall(M)",
}


def to_float(v):
    if v is None:
        return None
    try:
        x = float(str(v).strip())
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def fmt(v):
    return f"{v:.4f}" if v is not None else "NA"


def find_results_files():
    files = set()
    for root in ROOTS:
        if root.exists():
            for p in root.rglob("results.csv"):
                files.add(p.resolve())
    return sorted(files)


def extract_best_row(results_csv):
    try:
        with results_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None

            best_row = None
            best_score = -1.0

            for row in reader:
                score = to_float(row.get(METRIC_MAP["mask95"]))
                if score is None:
                    score = to_float(row.get(METRIC_MAP["box95"]))
                if score is None:
                    score = to_float(row.get(METRIC_MAP["mask50"]))

                if score is None:
                    continue

                if score > best_score:
                    best_score = score
                    best_row = row

            return best_row
    except Exception as e:
        print(f"skip parse error: {results_csv} ({e})")
        return None


def build_entries(results_files):
    rows = []

    for results_csv in results_files:
        run_dir = results_csv.parent
        best_row = extract_best_row(results_csv)
        if not best_row:
            continue

        epoch_val = to_float(best_row.get("epoch"))
        weights_dir = run_dir / "weights"

        rows.append({
            "run": str(run_dir),
            "epoch": int(epoch_val) if epoch_val is not None else -1,
            "best_pt": str(weights_dir / "best.pt"),
            "last_pt": str(weights_dir / "last.pt"),
            "box50": to_float(best_row.get(METRIC_MAP["box50"])),
            "box95": to_float(best_row.get(METRIC_MAP["box95"])),
            "mask50": to_float(best_row.get(METRIC_MAP["mask50"])),
            "mask95": to_float(best_row.get(METRIC_MAP["mask95"])),
            "p_box": to_float(best_row.get(METRIC_MAP["p_box"])),
            "r_box": to_float(best_row.get(METRIC_MAP["r_box"])),
            "p_mask": to_float(best_row.get(METRIC_MAP["p_mask"])),
            "r_mask": to_float(best_row.get(METRIC_MAP["r_mask"])),
        })

    return rows


def sort_runs(rows):
    return sorted(
        rows,
        key=lambda r: (
            r["mask95"] if r["mask95"] is not None else -1,
            r["box95"] if r["box95"] is not None else -1,
            r["mask50"] if r["mask50"] is not None else -1,
        ),
        reverse=True,
    )


def print_top_runs(rows):
    print("\n=== TOP RUNS (by mask mAP50-95) ===\n")
    for i, r in enumerate(rows[:20], 1):
        print(
            f"{i:2d}. {r['run']}\n"
            f"    epoch={r['epoch']}  "
            f"box50={fmt(r['box50'])}  "
            f"box95={fmt(r['box95'])}  "
            f"mask50={fmt(r['mask50'])}  "
            f"mask95={fmt(r['mask95'])}\n"
            f"    P(B)={fmt(r['p_box'])}  "
            f"R(B)={fmt(r['r_box'])}  "
            f"P(M)={fmt(r['p_mask'])}  "
            f"R(M)={fmt(r['r_mask'])}\n"
            f"    best={r['best_pt']}\n"
        )


def print_best(rows):
    best = rows[0]
    print("=== RECOMMENDED CHECKPOINT ===\n")
    print(f"Run:   {best['run']}")
    print(f"Epoch: {best['epoch']}")
    print(f"best.pt: {best['best_pt']}")
    print(f"last.pt: {best['last_pt']}")
    print(
        f"box50={fmt(best['box50'])}  "
        f"box95={fmt(best['box95'])}  "
        f"mask50={fmt(best['mask50'])}  "
        f"mask95={fmt(best['mask95'])}"
    )


def print_top_box(rows):
    print("\n=== TOP 10 (by box mAP50-95) ===\n")
    rows_box = sorted(
        rows,
        key=lambda r: r["box95"] if r["box95"] is not None else -1,
        reverse=True,
    )
    for i, r in enumerate(rows_box[:10], 1):
        print(
            f"{i:2d}. {r['run']} | epoch={r['epoch']} | "
            f"box95={fmt(r['box95'])} | mask95={fmt(r['mask95'])}"
        )


def main():
    results_files = find_results_files()

    if not results_files:
        print("No results.csv files found.")
        return

    rows = build_entries(results_files)

    if not rows:
        print("No valid runs found.")
        return

    rows_sorted = sort_runs(rows)
    print_top_runs(rows_sorted)
    print_best(rows_sorted)
    print_top_box(rows_sorted)


if __name__ == "__main__":
    main()