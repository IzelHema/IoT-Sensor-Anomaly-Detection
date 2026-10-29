"""Validation-only parameter sweep.

This script uses validation seeds 100-104, which are disjoint from main.py test
seeds 0-19. It writes ranked candidate settings but does NOT overwrite main.py.
Choose/freeze parameters before running the final test experiment.
"""
from pathlib import Path
import itertools
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic import generate_clean_series, random_events
from src.detectors import proposed_directional_detector
from src.evaluate import point_metrics, event_metrics

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(100, 105)
N = 6000

# Compact, defensible grid. Expand later if needed.
GRID = {
    "short_window": [20, 30, 50],
    "long_window": [200, 300, 500],
    "beta": [0.25, 0.50, 1.00],
    "drift_gate": [0.25, 0.50, 0.75, 1.00],
}
K = 3.0

rows = []
for sw, lw, beta, gate in itertools.product(
    GRID["short_window"], GRID["long_window"], GRID["beta"], GRID["drift_gate"]
):
    if lw <= sw * 3:
        continue
    fprs, event_recalls, f1s = [], [], []
    for seed in SEEDS:
        # Drift-only validation measures false alarms.
        drift_df, _ = generate_clean_series(
            n=N, seed=seed, drift_strength=4.0, drift_start=1200, drift_end=3600
        )
        x = drift_df["value"].to_numpy()
        y = drift_df["anomaly"].to_numpy()
        pred, *_ = proposed_directional_detector(x, sw, lw, K, beta, gate)
        valid = np.arange(N) >= lw
        fprs.append(point_metrics(y, pred, valid)["FPR"])

        # Drift+anomaly validation measures sensitivity.
        events = random_events(N, seed+700, count=24)
        da_df, _ = generate_clean_series(
            n=N, seed=seed, drift_strength=4.0, drift_start=1200, drift_end=3600,
            anomaly_events=events
        )
        x = da_df["value"].to_numpy(); y = da_df["anomaly"].to_numpy()
        pred, *_ = proposed_directional_detector(x, sw, lw, K, beta, gate)
        pm = point_metrics(y, pred, valid)
        em = event_metrics(y, pred)
        event_recalls.append(em["EventRecall"])
        f1s.append(pm["F1"])

    rows.append({
        "short_window": sw, "long_window": lw, "beta": beta, "drift_gate": gate,
        "FPR_mean": np.mean(fprs),
        "EventRecall_mean": np.mean(event_recalls),
        "F1_mean": np.mean(f1s),
    })

res = pd.DataFrame(rows)
# Rank by anomaly F1 first, then low FPR. Keep EventRecall visible to avoid hiding misses.
res = res.sort_values(["F1_mean", "FPR_mean"], ascending=[False, True])
res.to_csv(OUT / "validation_parameter_sweep.csv", index=False)
print(res.head(15).round(4).to_string(index=False))
print(f"\nSaved: {OUT / 'validation_parameter_sweep.csv'}")
