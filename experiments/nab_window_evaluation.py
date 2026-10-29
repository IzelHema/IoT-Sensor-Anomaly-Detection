"""Window-based event evaluation on the NAB machine-temperature series.

This is a transparent event/window evaluation, NOT the official NAB scoring
function. A window counts as detected if at least one prediction falls inside it.
False-positive points are predictions outside all labeled windows.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detectors import rolling_sigma_detector, proposed_directional_detector

DATA = ROOT / "data" / "real"
OUT = ROOT / "results" / "real_data"
OUT.mkdir(parents=True, exist_ok=True)
SERIES = DATA / "machine_temperature_system_failure.csv"
WINDOWS = DATA / "combined_windows.json"
KEY = "realKnownCause/machine_temperature_system_failure.csv"

PARAMS = {"short_window": 50, "long_window": 500, "k": 3.0,
          "beta": 0.25, "drift_gate": 0.75}

for p in (SERIES, WINDOWS):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}")

s = pd.read_csv(SERIES)
s["timestamp"] = pd.to_datetime(s["timestamp"])
s = s.sort_values("timestamp").reset_index(drop=True)
x = s["value"].to_numpy(dtype=float)
with open(WINDOWS, encoding="utf-8") as f:
    all_windows = json.load(f)
if KEY not in all_windows:
    raise KeyError(f"{KEY!r} not found in {WINDOWS}")
intervals = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in all_windows[KEY]]

# All detector decisions are produced using the same series and parameter k.
rolling = rolling_sigma_detector(x, PARAMS["short_window"], PARAMS["k"])[0].astype(bool)
proposed = proposed_directional_detector(
    x, PARAMS["short_window"], PARAMS["long_window"], PARAMS["k"],
    PARAMS["beta"], PARAMS["drift_gate"]
)[0].astype(bool)

# Create a point mask for descriptive point metrics and event-window metrics.
in_window = np.zeros(len(s), dtype=bool)
window_rows = []
for i, (start, end) in enumerate(intervals, 1):
    mask = s["timestamp"].between(start, end, inclusive="both").to_numpy()
    in_window |= mask
    window_rows.append({"window": i, "start": start, "end": end,
                        "labeled_readings": int(mask.sum())})


def summarize(name, pred):
    tp = int((pred & in_window).sum())
    fp = int((pred & ~in_window).sum())
    fn = int((~pred & in_window).sum())
    tn = int((~pred & ~in_window).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    detected = 0
    latencies = []
    event_detail = []
    for i, (start, end) in enumerate(intervals, 1):
        idx = np.flatnonzero(s["timestamp"].between(start, end, inclusive="both").to_numpy())
        hits = idx[pred[idx]] if len(idx) else np.array([], dtype=int)
        if len(hits):
            detected += 1
            first = int(hits[0])
            latency = first - int(idx[0])
            latencies.append(latency)
            first_ts = s.loc[first, "timestamp"]
        else:
            latency = np.nan
            first_ts = pd.NaT
        event_detail.append({"method": name, "window": i, "start": start, "end": end,
                             "detected": bool(len(hits)), "first_detection_timestamp": first_ts,
                             "latency_samples": latency, "predictions_in_window": int(len(hits))})

    summary = {
        "method": name, "total_flags": int(pred.sum()),
        "point_precision_window_labels": precision,
        "point_recall_window_labels": recall,
        "point_f1_window_labels": f1,
        "point_fpr_outside_windows": fpr,
        "false_positive_points_outside_windows": fp,
        "detected_windows": detected, "total_windows": len(intervals),
        "window_event_recall": detected / len(intervals) if intervals else np.nan,
        "mean_latency_samples_detected_windows": float(np.mean(latencies)) if latencies else np.nan,
        "median_latency_samples_detected_windows": float(np.median(latencies)) if latencies else np.nan,
    }
    return summary, event_detail

summaries, events = [], []
for name, pred in [("Rolling_kSigma", rolling), ("Proposed_Directional", proposed)]:
    summary, detail = summarize(name, pred)
    summaries.append(summary)
    events.extend(detail)

summary_df = pd.DataFrame(summaries)
summary_path = OUT / "nab_window_summary.csv"
event_path = OUT / "nab_window_event_details.csv"
summary_df.to_csv(summary_path, index=False)
pd.DataFrame(events).to_csv(event_path, index=False)
pd.DataFrame(window_rows).to_csv(OUT / "nab_labeled_windows.csv", index=False)

print("\n=== NAB WINDOW-BASED EVENT EVALUATION ===")
print("Series readings:", len(s))
print("Labeled anomaly windows:", len(intervals))
print("Parameters:", PARAMS)
print("\nMethod summary (point labels are derived from window membership):")
print(summary_df.round(4).to_string(index=False))
print("\nPer-window event detections:")
print(pd.DataFrame(events).to_string(index=False))
print("\nSaved:", summary_path)
print("Saved:", event_path)
print("NOTE: This is window-based evaluation, not the official NAB scoring algorithm.")
