# Research Method — Paper-Ready Experimental Design

## Working title
**Lightweight Direction-Aware Dual-Timescale Drift Compensation for Online IoT Sensor Anomaly Detection**

## Research question
Can a normalized, gated, direction-aware dual-timescale correction reduce false alarms caused by legitimate sensor drift without materially reducing anomaly sensitivity, while remaining suitable for microcontroller-class execution?

## Proposed detector
At time `t`, all statistics are computed from samples strictly before `t`.

Short-window statistics:

`mu_s(t), sigma_s(t)`

Long-window statistics:

`mu_l(t), sigma_l(t)`

Signed normalized drift:

`d(t) = (mu_s(t) - mu_l(t)) / (sigma_l(t) + epsilon)`

Directional compensation:

`C_plus(t)  = beta * sigma_s(t) * max(0, d(t) - delta)`

`C_minus(t) = beta * sigma_s(t) * max(0, -d(t) - delta)`

Thresholds:

`U(t) = mu_s(t) + k*sigma_s(t) + C_plus(t)`

`L(t) = mu_s(t) - k*sigma_s(t) - C_minus(t)`

An anomaly is declared if `x(t) > U(t)` or `x(t) < L(t)`.

## Why this version is used
- **Causal:** the current sample does not alter the threshold used to judge itself.
- **Normalized:** drift is measured relative to long-term variability.
- **Gated:** small harmless differences between windows do not trigger compensation.
- **Direction-aware:** positive drift only relaxes the upper boundary; negative drift only relaxes the lower boundary.
- **Lightweight:** only rolling means/variances and simple arithmetic are required.

## Baselines
1. Fixed two-sided k-sigma threshold, calibrated only on an initial normal prefix.
2. Causal rolling k-sigma detector.
3. Two-timescale drift-triggered recalibration baseline.
4. Proposed direction-aware dual-timescale detector.

All threshold methods use the same `k` in the default experiment for fairness.

## Synthetic ground truth
The controlled benchmark is generated from a clean normal stochastic process. Drift and anomalies are then injected separately, so every anomaly label is known exactly. No real dataset with hidden pre-existing anomalies is used as the synthetic background.

Default scenarios:
- stable
- gradual drift
- fast drift
- sudden anomalies
- drift + anomalies
- noisy drift

Multiple random seeds are used. Anomaly events include positive and negative deviations and multiple durations.

## Evaluation
### Drift/no-anomaly scenarios
Report:
- False Positive Rate (FPR)
- False alarms per 1000 normal samples

### Anomaly scenarios
Report:
- Precision
- Recall
- F1
- event-level recall
- event detection latency

Latency is counted only when detection occurs inside the true anomaly interval. A later unrelated false alarm is not counted as delayed detection.

## Validation/test separation
`experiments/parameter_sweep.py` uses validation seeds 100–104.
`main.py` uses disjoint test seeds 0–19.
Do not tune parameters after inspecting the test results.

## Ablation
`experiments/ablation.py` compares:
- ordinary rolling k-sigma
- symmetric drift compensation
- direction-aware compensation

This tests whether directionality itself contributes value.

## Anomaly-strength analysis
`experiments/anomaly_strength_sweep.py` measures performance over anomaly magnitudes of 2–6 standard deviations. This directly tests whether drift compensation hides weak anomalies.

## Real-data validation
The included NAB machine-temperature series is retained only as an optional real-world dataset. It must not be treated as a clean normal background. `experiments/nab_optional_validation.py` refuses to run unless official labels are separately supplied.

## Computational Scope and Limitations

This project implements and evaluates a lightweight, drift-aware anomaly detection method for IoT sensor streams using Python.

The proposed method uses short-term and long-term rolling statistics to adapt anomaly detection thresholds when the sensor baseline changes.

The experiments include synthetic sensor streams with different drift and anomaly conditions, parameter validation, ablation studies, statistical significance testing, and an exploratory evaluation using real-world sensor data from the Numenta Anomaly Benchmark (NAB).

All experiments were performed in a software environment. No physical IoT hardware, ESP32 implementation, or embedded-device testing was conducted.

Although the method uses simple statistical calculations, its actual memory consumption, execution time, and energy efficiency on embedded devices have not been measured. Hardware evaluation remains a possible direction for future work.
