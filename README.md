# Dual-Timescale Drift-Aware Anomaly Detection — Corrected Research Package

This folder is the corrected, reproducible version of the research prototype.

## What was fixed
- Removed the invalid practice of using a real anomaly-containing NAB series as a supposedly normal synthetic background.
- All online thresholds are now **causal**: `x_t` is never used to compute the threshold that judges `x_t`.
- Fixed detector is now two-sided and uses the same default `k` as other methods.
- Drift-recalibration baseline now adapts toward the recent regime instead of jumping back to the lagging long-term mean.
- Detection latency is event-bounded; unrelated later false alarms are never counted as delayed detections.
- No-anomaly false-alarm evaluation is separated from anomaly precision/recall/F1.
- Multiple anomaly events and multiple random seeds are used.
- Full and zoomed plots include the actual anomaly region.
- Added validation-only parameter sweep, ablation, anomaly-strength study, and optional properly labeled NAB validation.

## 1. Install
From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## 2. Recommended run order
### A. Optional parameter validation
```bash
python experiments/parameter_sweep.py
```
This uses validation seeds 100–104 only. Review:
`results/tables/validation_parameter_sweep.csv`

The default parameters in `main.py` are already frozen. If you decide to change them from the validation study, change them **before** running/finalizing the test experiment.

### B. Final controlled test experiment
```bash
python main.py
```
Main outputs:
- `results/tables/all_test_results.csv`
- `results/tables/false_alarm_summary.csv`
- `results/tables/anomaly_detection_summary.csv`
- `results/tables/scenario_summary.csv`
- `results/figures/*.png`
- `results/config/parameters_used.json`

### C. Ablation
```bash
python experiments/ablation.py
```
Outputs:
- `results/tables/ablation_all.csv`
- `results/tables/ablation_summary.csv`

### D. Anomaly-strength sensitivity
```bash
python experiments/anomaly_strength_sweep.py
```
Outputs:
- `results/tables/anomaly_strength_all.csv`
- `results/tables/anomaly_strength_summary.csv`

## 3. Optional NAB validation
The original NAB series is in:
`data/real/machine_temperature_system_failure.csv`

It is **not** used as a clean synthetic baseline because it contains real anomalies.
The optional script requires separately supplied official labels and will refuse to run without them:

```bash
python experiments/nab_optional_validation.py
```

See the script header for the required label-file format.

## 4. Paper method
Read `RESEARCH_METHOD.md`. It contains the exact equations, baselines, metrics, experimental logic, novelty explanation, computational characteristics, and research limitations.

## 5. Important research rule
Do not tune parameters on the test seeds after looking at final test results. Validation and test seeds are intentionally separated to avoid optimistic leakage.


## Real-World Evaluation — NAB Dataset

In addition to synthetic experiments, the proposed method was evaluated on the `machine_temperature_system_failure` sensor stream from the Numenta Anomaly Benchmark (NAB).

The dataset contains 22,695 timestamped observations and four labeled anomaly windows.

### Results

| Metric | Rolling k-sigma | Proposed Method |
|---|---:|---:|
| Total alerts | 636 | 593 |
| Anomaly windows detected | 4/4 | 4/4 |
| Point-level precision | 0.1447 | 0.1501 |
| Point-level recall | 0.0406 | 0.0392 |
| Point-level F1-score | 0.0634 | 0.0622 |

The proposed method generated **43 fewer alerts (6.76% reduction)** compared with rolling k-sigma.

However, point-level recall and F1-score decreased slightly. Therefore, the results indicate a reduction in alert volume rather than an overall improvement across every detection metric.

**Note:** This is an exploratory window-based evaluation and does not use the official NAB scoring procedure.

## Research Limitations

- The proposed method was evaluated using Python simulations and one real-world sensor stream.
- No physical IoT devices or embedded hardware were used.
- Execution time, RAM consumption, and energy efficiency were not experimentally measured.
- The method reduces false alarms under the evaluated conditions but does not consistently improve every anomaly detection metric.
- Further evaluation across different sensor types and real-world environments is required.

## Research Reproducibility

The repository includes the Python detector implementations, synthetic data generation scripts, experimental evaluation scripts, result tables, and figures.

Synthetic datasets can be regenerated using the provided scripts.

For experimental configurations and methodology, refer to `RESEARCH_METHOD.md` and `EXPERIMENTS.md`.