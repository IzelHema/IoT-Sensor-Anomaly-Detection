"""Optional NAB real-world validation helper.

This script intentionally does NOT invent labels. To obtain valid NAB results, place
an official point-label CSV named `machine_temperature_system_failure_labels.csv`
in data/real/ with columns: timestamp, anomaly (0/1). The timestamps must correspond
to the provided NAB series. If you prefer NAB's official anomaly-window scoring,
implement that scoring separately rather than converting windows to arbitrary points.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.detectors import rolling_sigma_detector, proposed_directional_detector
from src.evaluate import point_metrics, event_metrics

DATA = ROOT / "data" / "real"
series_path = DATA / "machine_temperature_system_failure.csv"
label_path = DATA / "machine_temperature_system_failure_labels.csv"
if not label_path.exists():
    raise FileNotFoundError(
        "Official labels are not bundled. Add data/real/machine_temperature_system_failure_labels.csv "
        "with timestamp,anomaly columns before using this optional validation."
    )

s = pd.read_csv(series_path)
l = pd.read_csv(label_path)
if "timestamp" not in s.columns or "value" not in s.columns:
    raise ValueError("NAB series must contain timestamp and value columns")

s["timestamp"] = pd.to_datetime(s["timestamp"], errors="raise")
l["timestamp"] = pd.to_datetime(l["timestamp"], errors="raise")
# Explicitly aggregate duplicate timestamps rather than silently keeping duplicates.
s = s.groupby("timestamp", as_index=False)["value"].mean()
l = l.groupby("timestamp", as_index=False)["anomaly"].max()
df = s.merge(l, on="timestamp", how="inner")
x = df["value"].to_numpy(); y = df["anomaly"].astype(int).to_numpy()
valid = np.arange(len(x)) >= 300

methods = {
    "Rolling": rolling_sigma_detector(x, 30, 3.0)[0],
    "Proposed_Directional": proposed_directional_detector(x, 30, 300, 3.0, 0.35, 1.5)[0],
}
for name, pred in methods.items():
    print("\n", name)
    print(point_metrics(y, pred, valid))
    print(event_metrics(y, pred))
