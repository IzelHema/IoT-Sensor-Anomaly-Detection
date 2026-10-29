import numpy as np

EPS = 1e-8


def _causal_rolling_stats(x, window, ddof=0):
    """Rolling mean/std using ONLY samples before time t.

    At index t, statistics are computed from x[max(0,t-window):t].
    Values remain NaN until `window` prior samples are available.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    if window < 2 or n <= window:
        return mu, sigma

    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    # Vectorized causal windows for t = window, ..., n-1.
    sums = cs[window:n] - cs[:n-window]
    sums2 = cs2[window:n] - cs2[:n-window]
    m = sums / window
    var = np.maximum(0.0, sums2 / window - m * m)
    if ddof == 1 and window > 1:
        var *= window / (window - 1)
    mu[window:n] = m
    sigma[window:n] = np.sqrt(var)
    return mu, sigma


def fixed_sigma_detector(x, calibration=500, k=3.0):
    """Two-sided fixed k-sigma detector calibrated on an initial normal prefix."""
    x = np.asarray(x, dtype=float)
    if calibration < 2 or calibration >= len(x):
        raise ValueError("calibration must be >=2 and smaller than series length")
    center = float(np.mean(x[:calibration]))
    scale = float(np.std(x[:calibration])) + EPS
    upper = np.full(len(x), center + k * scale)
    lower = np.full(len(x), center - k * scale)
    pred = (x > upper) | (x < lower)
    pred[:calibration] = False
    return pred, upper, lower


def rolling_sigma_detector(x, short_window=30, k=3.0):
    """Causal rolling k-sigma detector."""
    x = np.asarray(x, dtype=float)
    mu, sigma = _causal_rolling_stats(x, short_window)
    upper = mu + k * sigma
    lower = mu - k * sigma
    valid = np.isfinite(upper) & np.isfinite(lower)
    pred = valid & ((x > upper) | (x < lower))
    return pred, upper, lower


def drift_recalibration_detector(
    x, short_window=30, long_window=300, k=3.0,
    drift_gate=1.5, confirm_samples=8
):
    """Fair two-timescale baseline with explicit drift-triggered recalibration.

    Drift evidence is computed causally from short/long historical windows.
    When normalized mean shift exceeds the gate for `confirm_samples` consecutive
    decisions, the reference center is recalibrated to the short-window mean.
    Between recalibrations, the reference center remains fixed while the scale
    follows the causal short-window standard deviation.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    sm, ss = _causal_rolling_stats(x, short_window)
    lm, ls = _causal_rolling_stats(x, long_window)
    z = np.abs(sm - lm) / (ls + EPS)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    pred = np.zeros(n, dtype=bool)
    recalibrated = np.zeros(n, dtype=bool)

    start = long_window
    if n <= start:
        return pred, upper, lower, z, recalibrated

    ref_center = lm[start]
    run = 0
    for t in range(start, n):
        if not (np.isfinite(sm[t]) and np.isfinite(ss[t]) and np.isfinite(z[t])):
            continue

        if z[t] >= drift_gate:
            run += 1
        else:
            run = 0

        if run >= confirm_samples:
            ref_center = sm[t]
            recalibrated[t] = True
            run = 0

        scale = max(float(ss[t]), EPS)
        upper[t] = ref_center + k * scale
        lower[t] = ref_center - k * scale
        pred[t] = (x[t] > upper[t]) or (x[t] < lower[t])

    return pred, upper, lower, z, recalibrated


def proposed_directional_detector(
    x, short_window=30, long_window=300,
    k=3.0, beta=0.35, drift_gate=1.5
):
    """Proposed normalized, gated, direction-aware dual-timescale detector.

    d_t = (mu_short - mu_long) / (sigma_long + eps)
    C+  = beta * sigma_short * max(0, d_t - gate)
    C-  = beta * sigma_short * max(0, -d_t - gate)
    U_t = mu_short + k*sigma_short + C+
    L_t = mu_short - k*sigma_short - C-

    All statistics are causal and therefore exclude x_t itself.
    """
    x = np.asarray(x, dtype=float)
    sm, ss = _causal_rolling_stats(x, short_window)
    lm, ls = _causal_rolling_stats(x, long_window)

    d = (sm - lm) / (ls + EPS)
    c_up = beta * ss * np.maximum(0.0, d - drift_gate)
    c_down = beta * ss * np.maximum(0.0, -d - drift_gate)

    upper = sm + k * ss + c_up
    lower = sm - k * ss - c_down
    valid = np.isfinite(upper) & np.isfinite(lower)
    pred = valid & ((x > upper) | (x < lower))
    return pred, upper, lower, d, c_up, c_down


def proposed_symmetric_detector(
    x, short_window=30, long_window=300,
    k=3.0, beta=0.35, drift_gate=1.5
):
    """Symmetric ablation of the proposed detector."""
    x = np.asarray(x, dtype=float)
    sm, ss = _causal_rolling_stats(x, short_window)
    lm, ls = _causal_rolling_stats(x, long_window)
    z = np.abs(sm - lm) / (ls + EPS)
    c = beta * ss * np.maximum(0.0, z - drift_gate)
    upper = sm + k * ss + c
    lower = sm - k * ss - c
    valid = np.isfinite(upper) & np.isfinite(lower)
    pred = valid & ((x > upper) | (x < lower))
    return pred, upper, lower, z, c
