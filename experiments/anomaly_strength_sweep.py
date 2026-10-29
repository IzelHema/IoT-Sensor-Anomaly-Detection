"""Sensitivity to anomaly magnitude, expressed in normal-process sigma units."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synthetic import generate_clean_series, random_events, Event
from src.detectors import rolling_sigma_detector, proposed_directional_detector
from src.evaluate import point_metrics, event_metrics

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for mag in [2.0, 3.0, 4.0, 5.0, 6.0]:
    for seed in range(10):
        base_events = random_events(6000, seed+1200, count=24, magnitudes=(mag,))
        df, _ = generate_clean_series(
            n=6000, seed=seed, drift_strength=4.0, drift_start=1200, drift_end=3600,
            anomaly_events=base_events
        )
        x = df["value"].to_numpy(); y = df["anomaly"].to_numpy()
        methods = {
            "Rolling": rolling_sigma_detector(x, 50, 3.0)[0],
            "Proposed_Directional": proposed_directional_detector(x, 50, 500, 3.0, 0.25, 0.75)[0],
        }
        valid = np.arange(len(x)) >= 500
        for name, pred in methods.items():
            r = point_metrics(y, pred, valid); r.update(event_metrics(y, pred))
            r.update({"Magnitude_sigma": mag, "Seed": seed, "Method": name})
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / "anomaly_strength_all.csv", index=False)
summary = res.groupby(["Magnitude_sigma", "Method"], as_index=False).agg(
    F1_mean=("F1", "mean"), Recall_mean=("Recall", "mean"),
    EventRecall_mean=("EventRecall", "mean"), FPR_mean=("FPR", "mean")
)
summary.to_csv(OUT / "anomaly_strength_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
