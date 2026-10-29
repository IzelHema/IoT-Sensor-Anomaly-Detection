"""Tune rolling k-sigma baseline on validation seeds only (100-104).

Does not use the held-out test seeds 0-19 and does not change main.py.
Selects k using mean F1 on validation drift+anomaly cases; reports FPR too.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synthetic import generate_clean_series, random_events
from src.detectors import rolling_sigma_detector
from src.evaluate import point_metrics, event_metrics

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(100, 105)
N = 6000
SHORT_WINDOW = 50
LONG_WINDOW = 500
K_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
rows = []

for k in K_VALUES:
    drift_fprs = []
    f1s, recalls, event_recalls = [], [], []
    for seed in SEEDS:
        # Drift-only series measures false alarms under normal drift.
        drift_df, _ = generate_clean_series(
            n=N, seed=seed, drift_strength=4.0,
            drift_start=1200, drift_end=3600
        )
        dx = drift_df["value"].to_numpy(float)
        dp = rolling_sigma_detector(dx, SHORT_WINDOW, k)[0]
        valid = np.arange(N) >= LONG_WINDOW
        dm = point_metrics(drift_df["anomaly"].to_numpy(int), dp, valid)
        drift_fprs.append(dm["FPR"])

        # Disjoint deterministic event seeds for validation anomaly cases.
        events = random_events(N, seed + 700, count=24)
        da_df, _ = generate_clean_series(
            n=N, seed=seed, drift_strength=4.0,
            drift_start=1200, drift_end=3600, anomaly_events=events
        )
        x = da_df["value"].to_numpy(float)
        y = da_df["anomaly"].to_numpy(int)
        pred = rolling_sigma_detector(x, SHORT_WINDOW, k)[0]
        pm = point_metrics(y, pred, valid)
        em = event_metrics(y, pred)
        f1s.append(pm["F1"])
        recalls.append(pm["Recall"])
        event_recalls.append(em["EventRecall"])

    rows.append({
        "k": k,
        "validation_drift_FPR_mean": float(np.nanmean(drift_fprs)),
        "validation_F1_mean": float(np.nanmean(f1s)),
        "validation_recall_mean": float(np.nanmean(recalls)),
        "validation_event_recall_mean": float(np.nanmean(event_recalls)),
    })

res = pd.DataFrame(rows).sort_values(
    ["validation_F1_mean", "validation_drift_FPR_mean"],
    ascending=[False, True]
).reset_index(drop=True)
res["rank_by_F1_then_FPR"] = np.arange(1, len(res) + 1)
path = OUT / "baseline_k_validation_sweep.csv"
res.to_csv(path, index=False)
print("=== ROLLING BASELINE K SWEEP: VALIDATION SEEDS 100-104 ===")
print(res.round(4).to_string(index=False))
print("\nTop validation k:", res.iloc[0]["k"])
print("Saved:", path)
print("Do not describe this as test performance; run the frozen choice on held-out test seeds only once.")
