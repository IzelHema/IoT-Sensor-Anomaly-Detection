
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(
    ROOT / "results" / "real_data" / "real_data_predictions.csv"
)

# Readings flagged by rolling but not by proposed
removed = df[
    (df["rolling_anomaly"] == 1) &
    (df["proposed_anomaly"] == 0)
].copy()

removed["manual_review"] = ""
removed["notes"] = ""

output = ROOT / "results" / "real_data" / "removed_43_flags.csv"
removed.to_csv(output, index=False)

print("Removed flags:", len(removed))
print(removed[["timestamp", "value", "rolling_anomaly",
               "proposed_anomaly"]].to_string(index=False))
print("\nSaved to:", output)
