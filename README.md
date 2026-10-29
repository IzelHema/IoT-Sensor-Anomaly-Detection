# IoT Sensor Anomaly Detection Using Dual-Timescale Drift Compensation

**A lightweight, direction-aware statistical approach for reducing false alarms in continuously changing IoT sensor streams.**

## Overview

IoT sensors continuously generate data for applications such as industrial monitoring, environmental sensing, and equipment health monitoring. Detecting abnormal behavior in these streams is challenging because sensor readings can change naturally over time.

Traditional statistical anomaly detectors often use fixed or rolling thresholds. While these methods are computationally simple, changes in the sensor baseline can lead to unnecessary alerts.

This project implements a **direction-aware dual-timescale drift compensation method** that uses short-term and long-term rolling statistics to identify baseline changes and adjust anomaly detection thresholds accordingly.

The objective is to reduce false alarms caused by sensor drift while maintaining comparable anomaly detection performance.

The implementation and experiments are entirely software-based, using Python.

## Key Features

- Online anomaly detection using causal rolling statistics
- Short-term and long-term statistical monitoring
- Direction-aware threshold adjustment for upward and downward drift
- Comparison against fixed k-sigma, rolling k-sigma, and drift-recalibration baselines
- Evaluation across six synthetic sensor scenarios
- Parameter validation using separate random seeds
- Directional and symmetric compensation ablation
- Anomaly-strength sensitivity analysis
- Statistical significance testing
- Exploratory real-world evaluation using the Numenta Anomaly Benchmark (NAB)

## Proposed Method

The proposed detector uses two observation windows:

- **Short-term window:** Captures recent sensor behavior.
- **Long-term window:** Provides a reference for identifying changes in the underlying sensor baseline.

The difference between these statistics is used to estimate drift.

When upward drift is detected, the upper anomaly threshold is adjusted. When downward drift is detected, the lower threshold is adjusted.

This directional adjustment aims to reduce false alarms associated with baseline movement without unnecessarily widening both thresholds.

All thresholds are calculated using past observations, ensuring causal online evaluation.

For mathematical details, refer to `RESEARCH_METHOD.md`.

## Experimental Setup

The method was evaluated using six synthetic sensor scenarios:

| Scenario | Description |
|---|---|
| Stable | Sensor behavior without baseline drift |
| Gradual drift | Slowly changing sensor baseline |
| Fast drift | Rapid baseline movement |
| Sudden anomalies | Short abnormal events |
| Drift with anomalies | Anomalies occurring during baseline changes |
| Noisy drift | Baseline movement combined with increased noise |

Four detection approaches were compared:

1. Fixed k-sigma
2. Rolling k-sigma
3. Drift recalibration
4. Proposed directional dual-timescale detector

The main evaluation used 20 held-out random seeds, with separate validation seeds used for parameter selection.

## Results

### False-Positive Rate

The following results summarize the no-anomaly synthetic scenarios.

| Method | Mean False-Positive Rate |
|---|---:|
| Fixed k-sigma | 0.580086 |
| Drift recalibration | 0.012914 |
| Rolling k-sigma | 0.005407 |
| Proposed directional method | **0.005250** |

The proposed method achieved a **2.90% relative reduction in false-positive rate** compared with rolling k-sigma.

### Anomaly Detection Performance

| Method | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fixed k-sigma | 0.3695 | 0.8104 | 0.3862 |
| Drift recalibration | 0.3638 | 0.6732 | 0.4656 |
| Rolling k-sigma | 0.6463 | 0.6606 | **0.6519** |
| Proposed directional method | **0.6474** | 0.6586 | 0.6516 |

The proposed detector achieved slightly higher precision and lower false-positive rate than rolling k-sigma, with small reductions in recall and F1-score.

These results suggest that drift compensation can reduce false alarms, although it does not consistently improve all detection metrics.

## Real-World Evaluation

The proposed method was also evaluated using the `machine_temperature_system_failure` sensor series from the Numenta Anomaly Benchmark (NAB).

The dataset contains 22,695 timestamped observations and four labeled anomaly windows.

| Metric | Rolling k-sigma | Proposed Method |
|---|---:|---:|
| Total alerts | 636 | **593** |
| Anomaly windows detected | 4/4 | 4/4 |
| Point-level precision | 0.1447 | 0.1501 |
| Point-level recall | 0.0406 | 0.0392 |
| Point-level F1-score | 0.0634 | 0.0622 |

The proposed detector generated **43 fewer alerts (6.76% reduction)** while identifying all four labeled anomaly windows under the window-hit criterion.

However, point-level recall and F1-score decreased slightly.

**Note:** This experiment uses exploratory window-based evaluation and does not represent official NAB benchmark scoring.

## Installation

### Requirements

- Python
- Dependencies listed in `requirements.txt`

Clone the repository:

```bash
git clone https://github.com/IzelHema/IoT-Sensor-Anomaly-Detection.git
cd IoT-Sensor-Anomaly-Detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Main Evaluation

```bash
python main.py
```

This generates the synthetic benchmark results and visualizations.

### Parameter Validation

```bash
python experiments/parameter_sweep.py
```

### Directional and Symmetric Ablation

```bash
python experiments/ablation.py
```

### Anomaly-Strength Sensitivity

```bash
python experiments/anomaly_strength_sweep.py
```

### Baseline Threshold Sensitivity

```bash
python experiments/baseline_k_sweep.py
```

The generated outputs are stored under `results/`.

For additional experiment details, refer to `EXPERIMENTS.md`.

## Dataset Information

Synthetic sensor streams are generated programmatically and can be recreated using the provided implementation.

The real-world evaluation uses the machine-temperature sensor series from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB).

The original NAB input dataset is not included in this repository. Users should obtain the data and corresponding anomaly-window labels from the original source.

The real-data evaluation scripts are available under `experiments/`.

## Repository Contents

```text
IoT-Sensor-Anomaly-Detection/
├── README.md
├── RESEARCH_METHOD.md
├── EXPERIMENTS.md
├── requirements.txt
├── main.py
├── run_real_data.py
├── check_43_flags.py
├── experiments/
├── src/
├── results/
│   ├── config/
│   ├── figures/
│   ├── real_data/
│   └── tables/
├── RUN_ALL_EXPERIMENTS.bat
└── RUN_FINAL.bat
```

## Research Limitations

- Experiments were conducted using Python, without physical IoT hardware.
- No experimental measurements of RAM usage, execution time, or energy consumption were performed.
- Real-world evaluation was limited to one NAB sensor stream.
- Lower false-positive rates do not necessarily imply improved recall or F1-score.
- Further validation across additional sensor types and operational environments is needed.

## Documentation

- `RESEARCH_METHOD.md` — Mathematical formulation, detector design, baselines, and methodology.
- `EXPERIMENTS.md` — Experimental scenarios, evaluation setup, and results.
- `results/tables/` — Saved numerical results.
- `results/figures/` — Synthetic experiment visualizations.
- `results/real_data/` — Real-data evaluation outputs.

## Project Status

Research prototype implemented and evaluated using synthetic and exploratory real-world sensor data.

Further work may include testing on additional real-world datasets and benchmarking computational performance on embedded IoT devices.