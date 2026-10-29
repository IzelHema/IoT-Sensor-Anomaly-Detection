from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Event:
    start: int
    end: int   # exclusive
    magnitude_sigma: float


def _ar1_noise(n, sigma, phi, rng):
    e = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + e[i]
    return x


def generate_clean_series(
    n=6000, seed=0, base_level=25.0, noise_std=0.35,
    ar_phi=0.25, seasonal_amp=0.15, seasonal_period=720,
    drift_strength=0.0, drift_start=1200, drift_end=3600,
    extra_noise_std=0.0, anomaly_events=None
):
    """Generate a clean normal process, then separately add drift and labeled anomalies.

    The returned labels are exact because the normal process is synthetic and contains
    no hidden pre-existing anomaly labels.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    normal = base_level + _ar1_noise(n, noise_std, ar_phi, rng)
    normal += seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)

    drift = np.zeros(n)
    if drift_strength != 0.0 and drift_end > drift_start:
        s = max(0, min(drift_start, n))
        e = max(s, min(drift_end, n))
        if e > s:
            drift[s:e] = np.linspace(0.0, drift_strength, e-s, endpoint=False)
        if e < n:
            drift[e:] = drift_strength

    x = normal + drift
    if extra_noise_std > 0:
        x = x + rng.normal(0.0, extra_noise_std, n)

    labels = np.zeros(n, dtype=int)
    event_id = np.full(n, -1, dtype=int)
    events = []
    anomaly_events = anomaly_events or []
    for eid, ev in enumerate(anomaly_events):
        s = max(0, min(int(ev.start), n))
        e = max(s, min(int(ev.end), n))
        if e <= s:
            continue
        # Magnitude is expressed in base noise standard deviations for reproducibility.
        mag = float(ev.magnitude_sigma) * noise_std
        x[s:e] += mag
        labels[s:e] = 1
        event_id[s:e] = eid
        events.append({"event_id": eid, "start": s, "end": e,
                       "magnitude_sigma": float(ev.magnitude_sigma)})

    df = pd.DataFrame({
        "sample": t,
        "value": x,
        "normal_component": normal,
        "drift_component": drift,
        "anomaly": labels,
        "event_id": event_id,
    })
    return df, pd.DataFrame(events)


def random_events(n, seed, count=24, start_after=800, min_gap=80,
                  durations=(1, 2, 3, 5), magnitudes=(3.0, 4.0, 5.0, 6.0)):
    """Create separated positive and negative anomaly events."""
    rng = np.random.default_rng(seed)
    candidates = np.arange(start_after, n-20)
    rng.shuffle(candidates)
    starts = []
    for c in candidates:
        if all(abs(c-s) >= min_gap for s in starts):
            starts.append(int(c))
        if len(starts) == count:
            break
    starts.sort()
    events = []
    for s in starts:
        dur = int(rng.choice(durations))
        mag = float(rng.choice(magnitudes))
        if rng.random() < 0.5:
            mag = -mag
        events.append(Event(s, min(s+dur, n), mag))
    return events


def scenario_specs(n=6000, seed=0):
    return {
        "stable": dict(),
        "gradual_drift": dict(drift_strength=4.0, drift_start=1200, drift_end=3600),
        "fast_drift": dict(drift_strength=4.0, drift_start=1200, drift_end=1650),
        "sudden_anomalies": dict(anomaly_events=random_events(n, seed+101)),
        "drift_plus_anomalies": dict(
            drift_strength=4.0, drift_start=1200, drift_end=3600,
            anomaly_events=random_events(n, seed+202)
        ),
        "noisy_drift": dict(
            drift_strength=4.0, drift_start=1200, drift_end=3600,
            extra_noise_std=0.25
        ),
    }


def create_experiment_set(out_dir, n=6000, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for name, kwargs in scenario_specs(n=n, seed=seed).items():
        df, events = generate_clean_series(n=n, seed=seed, **kwargs)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        events.to_csv(out_dir / f"{name}_events.csv", index=False)
        created.append(name)
    return created
