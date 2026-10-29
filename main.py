from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.synthetic import create_experiment_set
from src.detectors import (
    fixed_sigma_detector, rolling_sigma_detector,
    drift_recalibration_detector, proposed_directional_detector,
    proposed_symmetric_detector,
)
from src.evaluate import point_metrics, event_metrics, summarize_results

ROOT = Path(__file__).resolve().parent
SYN = ROOT / "data" / "synthetic"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
CFG = ROOT / "results" / "config"
for p in (SYN, FIG, TAB, CFG):
    p.mkdir(parents=True, exist_ok=True)

# Pre-registered/default parameters. Use experiments/parameter_sweep.py to tune on
# validation seeds only; never tune on the test seeds used below.
PARAMS = {
    "calibration": 500,
    "short_window": 50,
    "long_window": 500,
    "k": 3.0,
    "beta": 0.25,
    "drift_gate": 0.75,
    "confirm_samples": 8,
}
TEST_SEEDS = list(range(20))
N = 6000

with open(CFG / "parameters_used.json", "w", encoding="utf-8") as f:
    json.dump({**PARAMS, "test_seeds": TEST_SEEDS, "series_length": N}, f, indent=2)


def run_methods(x):
    p = PARAMS
    f = fixed_sigma_detector(x, p["calibration"], p["k"])
    r = rolling_sigma_detector(x, p["short_window"], p["k"])
    d = drift_recalibration_detector(
        x, p["short_window"], p["long_window"], p["k"],
        p["drift_gate"], p["confirm_samples"]
    )
    pr = proposed_directional_detector(
        x, p["short_window"], p["long_window"], p["k"],
        p["beta"], p["drift_gate"]
    )
    return {
        "Fixed_kSigma": (f[0], f[1], f[2]),
        "Rolling_kSigma": (r[0], r[1], r[2]),
        "Drift_Recalibration": (d[0], d[1], d[2]),
        "Proposed_Directional": (pr[0], pr[1], pr[2]),
    }


rows = []
for seed in TEST_SEEDS:
    seed_dir = SYN / f"seed_{seed:02d}"
    names = create_experiment_set(seed_dir, n=N, seed=seed)
    for name in names:
        df = pd.read_csv(seed_dir / f"{name}.csv")
        x = df["value"].to_numpy(float)
        y = df["anomaly"].to_numpy(int)
        methods = run_methods(x)
        valid = np.arange(len(x)) >= PARAMS["long_window"]

        for method, (pred, upper, lower) in methods.items():
            m = point_metrics(y, pred, valid_mask=valid)
            m.update(event_metrics(y, pred))
            m.update({
                "Seed": seed,
                "Scenario": name,
                "Method": method,
                "HasAnomalies": bool(y.sum() > 0),
            })
            rows.append(m)

        # One representative set of paper-ready figures from seed 0.
        if seed == 0:
            pr_pred, pr_u, pr_l = methods["Proposed_Directional"]
            ro_pred, ro_u, ro_l = methods["Rolling_kSigma"]
            t = np.arange(len(x))
            plt.figure(figsize=(12, 5))
            plt.plot(t, x, linewidth=0.8, label="Sensor value")
            plt.plot(t, ro_u, linewidth=0.9, label="Rolling upper")
            plt.plot(t, pr_u, linewidth=0.9, label="Proposed upper")
            idx = np.where(y == 1)[0]
            if len(idx):
                plt.scatter(idx, x[idx], s=14, label="Ground-truth anomaly")
            plt.title(f"{name.replace('_', ' ').title()} — causal thresholds")
            plt.xlabel("Sample")
            plt.ylabel("Sensor value")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(FIG / f"{name}_full.png", dpi=220)
            plt.close()

            if len(idx):
                lo = max(0, idx[0] - 120)
                hi = min(len(x), idx[0] + 180)
                zt = np.arange(lo, hi)
                plt.figure(figsize=(12, 5))
                plt.plot(zt, x[lo:hi], linewidth=1.0, label="Sensor value")
                plt.plot(zt, ro_u[lo:hi], label="Rolling upper")
                plt.plot(zt, ro_l[lo:hi], label="Rolling lower")
                plt.plot(zt, pr_u[lo:hi], label="Proposed upper")
                plt.plot(zt, pr_l[lo:hi], label="Proposed lower")
                zi = idx[(idx >= lo) & (idx < hi)]
                plt.scatter(zi, x[zi], s=22, label="Ground-truth anomaly")
                plt.title(f"{name.replace('_', ' ').title()} — anomaly zoom")
                plt.xlabel("Sample")
                plt.ylabel("Sensor value")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(FIG / f"{name}_zoom.png", dpi=220)
                plt.close()

results = pd.DataFrame(rows)
results.to_csv(TAB / "all_test_results.csv", index=False)
false_alarm, detection = summarize_results(results)
false_alarm.to_csv(TAB / "false_alarm_summary.csv", index=False)
detection.to_csv(TAB / "anomaly_detection_summary.csv", index=False)

# Per-scenario tables are useful for the Results section.
scenario_summary = results.groupby(["Scenario", "Method"], as_index=False).agg(
    FPR_mean=("FPR", "mean"),
    FPR_std=("FPR", "std"),
    Precision_mean=("Precision", "mean"),
    Recall_mean=("Recall", "mean"),
    F1_mean=("F1", "mean"),
    EventRecall_mean=("EventRecall", "mean"),
    Latency_mean=("MeanLatency_samples", "mean"),
)
scenario_summary.to_csv(TAB / "scenario_summary.csv", index=False)

print("\n=== FALSE-ALARM PERFORMANCE (NO-ANOMALY SCENARIOS) ===")
print(false_alarm.round(4).to_string(index=False))
print("\n=== ANOMALY-DETECTION PERFORMANCE ===")
print(detection.round(4).to_string(index=False))
print(f"\nDetailed results: {TAB / 'all_test_results.csv'}")
print(f"Figures:          {FIG}")
