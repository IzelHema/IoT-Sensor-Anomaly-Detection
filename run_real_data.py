
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.detectors import (
    rolling_sigma_detector,
    proposed_directional_detector,
)


# =========================================================
# 1. PROJECT DIRECTORIES
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "real"
RESULTS = ROOT / "results" / "real_data"

RESULTS.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. DETECTOR PARAMETERS
# =========================================================

# Keep these consistent with the research experiment.
PARAMS = {
    "short_window": 50,
    "long_window": 500,
    "k": 3.0,
    "beta": 0.25,
    "drift_gate": 0.75,
}


# =========================================================
# 3. LOAD REAL DATASET
# =========================================================

data_path = DATA / "machine_temperature_system_failure.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

df = pd.read_csv(data_path)

if not {"timestamp", "value"}.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'timestamp' and 'value' columns."
    )

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

values = df["value"].to_numpy(dtype=float)

print("\n=== REAL DATA EXPERIMENT ===")
print(f"Total readings: {len(df)}")
print(f"Parameters: {PARAMS}")


# =========================================================
# 4. RUN ROLLING SIGMA BASELINE
# =========================================================

rolling_pred, rolling_upper, rolling_lower = (
    rolling_sigma_detector(
        values,
        short_window=PARAMS["short_window"],
        k=PARAMS["k"],
    )
)


# =========================================================
# 5. RUN PROPOSED DIRECTIONAL DETECTOR
# =========================================================

(
    proposed_pred,
    proposed_upper,
    proposed_lower,
    drift,
    correction_upper,
    correction_lower,
) = proposed_directional_detector(
    values,
    short_window=PARAMS["short_window"],
    long_window=PARAMS["long_window"],
    k=PARAMS["k"],
    beta=PARAMS["beta"],
    drift_gate=PARAMS["drift_gate"],
)


# =========================================================
# 6. SAVE PREDICTIONS
# =========================================================

df["rolling_anomaly"] = np.asarray(
    rolling_pred, dtype=int
)

df["proposed_anomaly"] = np.asarray(
    proposed_pred, dtype=int
)

df["rolling_upper"] = rolling_upper
df["rolling_lower"] = rolling_lower

df["proposed_upper"] = proposed_upper
df["proposed_lower"] = proposed_lower

df["normalized_drift"] = drift

print(
    f"Rolling detector flags: "
    f"{int(df['rolling_anomaly'].sum())}"
)

print(
    f"Proposed detector flags: "
    f"{int(df['proposed_anomaly'].sum())}"
)

print(
    "Rolling-only flags:",
    int(
        (
            (df["rolling_anomaly"] == 1)
            & (df["proposed_anomaly"] == 0)
        ).sum()
    )
)

print(
    "Proposed-only flags:",
    int(
        (
            (df["rolling_anomaly"] == 0)
            & (df["proposed_anomaly"] == 1)
        ).sum()
    )
)


# =========================================================
# 7. PLOT REAL SENSOR DATA AND DETECTIONS
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["timestamp"],
    values,
    linewidth=0.8,
    label="Real sensor values",
)

plt.plot(
    df["timestamp"],
    rolling_upper,
    linewidth=0.8,
    label="Rolling upper threshold",
)

plt.plot(
    df["timestamp"],
    proposed_upper,
    linewidth=0.9,
    label="Proposed upper threshold",
)

plt.plot(
    df["timestamp"],
    proposed_lower,
    linewidth=0.9,
    label="Proposed lower threshold",
)

proposed_mask = df["proposed_anomaly"] == 1

plt.scatter(
    df.loc[proposed_mask, "timestamp"],
    df.loc[proposed_mask, "value"],
    s=12,
    label="Proposed anomaly flags",
)

plt.title("Real Machine Temperature Anomaly Detection")
plt.xlabel("Timestamp")
plt.ylabel("Sensor value")
plt.legend()
plt.tight_layout()

plot_path = RESULTS / "real_data_anomalies.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"\nPlot saved to: {plot_path}")


# =========================================================
# 8. LOAD OFFICIAL NAB ANOMALY WINDOWS
# =========================================================

labels_path = DATA / "combined_windows.json"

if not labels_path.exists():
    print("\nWARNING: Official anomaly windows file not found.")
    print(f"Expected location: {labels_path}")
    print("Predictions and plot have still been saved.")

else:
    with open(labels_path, "r", encoding="utf-8") as file:
        all_windows = json.load(file)

    label_key = (
        "realKnownCause/"
        "machine_temperature_system_failure.csv"
    )

    if label_key not in all_windows:
        print("\nERROR: Dataset label key was not found.")
        print(
            "Available matching keys:",
            [
                key for key in all_windows
                if "machine_temperature" in key
            ],
        )

    else:
        intervals = all_windows[label_key]

        # Mark timestamps inside official anomaly windows.
        # These are window-derived point labels.
        df["anomaly"] = 0

        for start, end in intervals:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)

            in_window = df["timestamp"].between(
                start,
                end,
                inclusive="both",
            )

            df.loc[in_window, "anomaly"] = 1

        y = df["anomaly"].astype(int)

        print("\n=== OFFICIAL ANOMALY WINDOWS ===")
        print(f"Number of labeled windows: {len(intervals)}")
        print(
            "Readings inside labeled windows:",
            int(y.sum()),
        )
        print(
            "Readings outside labeled windows:",
            int((y == 0).sum()),
        )


        # =================================================
        # 9. CHECK THE FLAGS REMOVED BY THE PROPOSED METHOD
        # =================================================

        removed_flags = (
            (df["rolling_anomaly"] == 1)
            & (df["proposed_anomaly"] == 0)
        )

        outside_windows = int(
            (removed_flags & (y == 0)).sum()
        )

        inside_windows = int(
            (removed_flags & (y == 1)).sum()
        )

        print("\n=== CHECKING REMOVED FLAGS ===")

        print(
            "Removed flags outside labeled anomaly windows:",
            outside_windows,
        )

        print(
            "Removed flags inside labeled anomaly windows:",
            inside_windows,
        )

        print(
            "Total removed flags:",
            int(removed_flags.sum()),
        )


        # =================================================
        # 10. EVALUATE BOTH DETECTORS
        # =================================================

        for name, column in [
            ("Rolling detector", "rolling_anomaly"),
            ("Proposed detector", "proposed_anomaly"),
        ]:
            pred = df[column].astype(int)

            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            fn = int(((pred == 0) & (y == 1)).sum())
            tn = int(((pred == 0) & (y == 0)).sum())

            precision = (
                tp / (tp + fp)
                if tp + fp else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn else 0.0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            fpr = (
                fp / (fp + tn)
                if fp + tn else 0.0
            )

            print(f"\n=== {name.upper()} ===")
            print(f"TP: {tp}")
            print(f"FP: {fp}")
            print(f"FN: {fn}")
            print(f"TN: {tn}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"FPR:       {fpr:.4f}")


        # ================================================
        # 11. SAVE LABELED PREDICTIONS
        # ================================================

        labeled_output = RESULTS / "real_data_labeled_predictions.csv"

        df.to_csv(labeled_output, index=False)

        print(
            "\nLabeled predictions saved to:",
            labeled_output,
        )


# =========================================================
# 12. SAVE THE PREDICTION CSV
# =========================================================

output_csv = RESULTS / "real_data_predictions.csv"

df.to_csv(output_csv, index=False)

print("\n=== EXPERIMENT COMPLETE ===")
print(f"Predictions CSV: {output_csv}")
print(f"Plot: {plot_path}")
