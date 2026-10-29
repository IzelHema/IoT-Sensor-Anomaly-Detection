import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def point_metrics(y_true, y_pred, valid_mask=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    mask = np.ones(len(y_true), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    yt, yp = y_true[mask], y_pred[mask]
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    normal_n = tn + fp
    anomaly_n = tp + fn
    out = {
        "Precision": precision_score(yt, yp, zero_division=0) if anomaly_n else np.nan,
        "Recall": recall_score(yt, yp, zero_division=0) if anomaly_n else np.nan,
        "F1": f1_score(yt, yp, zero_division=0) if anomaly_n else np.nan,
        "FPR": fp / normal_n if normal_n else np.nan,
        "FNR": fn / anomaly_n if anomaly_n else np.nan,
        "FalseAlarms_per_1000": 1000.0 * fp / normal_n if normal_n else np.nan,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "NormalSamples": int(normal_n), "AnomalySamples": int(anomaly_n),
    }
    return out


def extract_events(y_true):
    y = np.asarray(y_true, dtype=int)
    events = []
    in_event = False
    start = None
    for i, v in enumerate(y):
        if v == 1 and not in_event:
            start, in_event = i, True
        if in_event and (v == 0 or i == len(y)-1):
            end = i if v == 0 else i+1
            events.append((start, end))
            in_event = False
    return events


def event_metrics(y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=int)
    events = extract_events(y_true)
    if not events:
        return {
            "EventCount": 0, "DetectedEvents": 0, "EventRecall": np.nan,
            "MeanLatency_samples": np.nan, "MedianLatency_samples": np.nan,
            "MissedEvents": 0,
        }
    latencies = []
    detected = 0
    for start, end in events:
        hits = np.where(y_pred[start:end] == 1)[0]
        if len(hits):
            detected += 1
            latencies.append(int(hits[0]))
    missed = len(events) - detected
    return {
        "EventCount": len(events),
        "DetectedEvents": detected,
        "EventRecall": detected / len(events),
        "MeanLatency_samples": float(np.mean(latencies)) if latencies else np.nan,
        "MedianLatency_samples": float(np.median(latencies)) if latencies else np.nan,
        "MissedEvents": missed,
    }


def summarize_results(df):
    """Separate no-anomaly false-alarm behavior from anomaly-detection behavior."""
    no_anom = df[df["HasAnomalies"] == False]
    with_anom = df[df["HasAnomalies"] == True]
    false_alarm = no_anom.groupby("Method", as_index=False).agg(
        FPR_mean=("FPR", "mean"),
        FPR_std=("FPR", "std"),
        FalseAlarms_per_1000_mean=("FalseAlarms_per_1000", "mean"),
        FalseAlarms_per_1000_std=("FalseAlarms_per_1000", "std"),
    )
    detection = with_anom.groupby("Method", as_index=False).agg(
        Precision_mean=("Precision", "mean"),
        Precision_std=("Precision", "std"),
        Recall_mean=("Recall", "mean"),
        Recall_std=("Recall", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        EventRecall_mean=("EventRecall", "mean"),
        EventRecall_std=("EventRecall", "std"),
        MeanLatency_samples_mean=("MeanLatency_samples", "mean"),
        MeanLatency_samples_std=("MeanLatency_samples", "std"),
    )
    return false_alarm, detection
