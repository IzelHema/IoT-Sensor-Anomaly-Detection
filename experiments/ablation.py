"""Ablation: rolling baseline vs symmetric correction vs directional correction."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synthetic import generate_clean_series, random_events
from src.detectors import rolling_sigma_detector, proposed_symmetric_detector, proposed_directional_detector
from src.evaluate import point_metrics, event_metrics

OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for seed in range(20):
    events = random_events(6000, seed+900, count=24)
    df, _ = generate_clean_series(
        n=6000, seed=seed, drift_strength=4.0, drift_start=1200, drift_end=3600,
        anomaly_events=events
    )
    x = df["value"].to_numpy(); y = df["anomaly"].to_numpy()
    methods = {
        "Rolling": rolling_sigma_detector(x, 50, 3.0)[0],
        "Symmetric": proposed_symmetric_detector(x, 50, 500, 3.0, 0.25, 0.75)[0],
        "Directional": proposed_directional_detector(x, 50, 500, 3.0, 0.25, 0.75)[0],
    }
    valid = np.arange(len(x)) >= 500
    for name, pred in methods.items():
        r = point_metrics(y, pred, valid)
        r.update(event_metrics(y, pred))
        r.update({"Seed": seed, "Method": name})
        rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(OUT / "ablation_all.csv", index=False)
summary = res.groupby("Method", as_index=False).agg(
    FPR_mean=("FPR", "mean"), F1_mean=("F1", "mean"),
    EventRecall_mean=("EventRecall", "mean"),
    Latency_mean=("MeanLatency_samples", "mean")
)
summary.to_csv(OUT / "ablation_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
