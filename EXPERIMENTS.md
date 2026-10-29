# Experiments and Evaluation

## 1. Project Overview

This project evaluates a lightweight, direction-aware dual-timescale drift compensation method for online IoT sensor anomaly detection.

The objective is to reduce false alarms caused by changes in sensor baselines while maintaining anomaly detection performance.

## 2. Synthetic Data Experiments

The proposed method was evaluated using six synthetic sensor scenarios:

1. Stable sensor behavior
2. Gradual baseline drift
3. Fast baseline drift
4. Sudden anomalies
5. Drift combined with anomalies
6. Noisy drift

The experiments used 20 held-out test seeds (0–19) to evaluate performance across different generated streams.

## 3. Compared Detection Methods

Four methods were evaluated:

- Fixed k-sigma threshold
- Rolling k-sigma threshold
- Drift recalibration baseline
- Proposed directional dual-timescale detector

## 4. Parameter Validation

Parameter selection was performed using separate validation seeds (100–104).

The final configuration was:

| Parameter | Value |
|---|---|
| Short-term window | 50 |
| Long-term window | 500 |
| Threshold multiplier (k) | 3.0 |
| Drift compensation factor (beta) | 0.25 |
| Drift gate (delta) | 0.75 |
| Calibration samples | 500 |

The selected configuration was then used for held-out testing.

## 5. Additional Experiments

The project includes:

- Parameter sensitivity analysis
- Baseline threshold sensitivity analysis
- Directional versus symmetric compensation ablation
- Anomaly-strength sensitivity experiments
- Paired statistical significance testing

These experiments examine false-positive behavior, anomaly detection performance, and the effect of different parameter choices.

## 6. Real-World Evaluation

An exploratory real-data experiment was performed using the NAB machine-temperature sensor dataset.

The rolling k-sigma detector generated 636 alerts, while the proposed detector generated 593 alerts, representing a 6.76% reduction.

Both methods detected all four labeled anomaly windows under the window-hit criterion.

This evaluation is not equivalent to official NAB benchmark scoring.

## 7. Evaluation Metrics

The evaluation includes:

- False-positive rate (FPR)
- Precision
- Recall
- F1-score
- Event-level recall
- Detection latency
- Anomaly-window detection coverage

## 8. Research Limitations

The experiments were conducted in Python without physical IoT hardware.

The real-data evaluation covers one NAB sensor stream, so the findings should not be generalized to all IoT applications.

Lower false-positive rates do not necessarily imply improvements in recall or F1-score.

No embedded-device processing time, RAM consumption, or energy measurements were performed.