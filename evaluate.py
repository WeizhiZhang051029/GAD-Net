"""
Evaluation Module
Computes all metrics: MAE, MSE, RMSE, MAPE, R2, CORR, HR
"""
import numpy as np
import config


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def compute_rmse(y_true, y_pred):
    return np.sqrt(compute_mse(y_true, y_pred))


def compute_mape(y_true, y_pred, epsilon=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def compute_corr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    centered_true = y_true - np.mean(y_true)
    centered_pred = y_pred - np.mean(y_pred)
    denominator = np.sqrt(np.sum(centered_true ** 2) * np.sum(centered_pred ** 2))
    if denominator == 0:
        return 0.0
    r = np.sum(centered_true * centered_pred) / denominator
    return float(r) if np.isfinite(r) else 0.0


def compute_hit_rate(y_true, y_pred, threshold=None):
    """
    Hit Rate: percentage of samples where |pred - true| < threshold.
    Following Eq. 16 in reference paper.
    """
    if threshold is None:
        threshold = config.HIT_RATE_THRESHOLD
    hits = np.abs(y_true - y_pred) < threshold
    return np.mean(hits) * 100


# Report overall metrics and metrics on the training-defined tail region.
def compute_all_metrics(y_true, y_pred, tail_bounds=None):
    """
    Compute all evaluation metrics, INCLUDING TAIL PERFORMANCE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {
        "MAE": compute_mae(y_true, y_pred),
        "MSE": compute_mse(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
        "CORR": compute_corr(y_true, y_pred),
        "HR": compute_hit_rate(y_true, y_pred),
    }


    if tail_bounds is None:
        lower_q = np.percentile(y_true, 10)
        upper_q = np.percentile(y_true, 90)
    else:
        lower_q, upper_q = map(float, tail_bounds)


    # Use bounds fitted on the training fold when they are supplied.
    tail_mask = (y_true <= lower_q) | (y_true >= upper_q)


    if np.sum(tail_mask) > 0:
        y_true_tail = y_true[tail_mask]
        y_pred_tail = y_pred[tail_mask]


        metrics["Tail_HR"] = compute_hit_rate(y_true_tail, y_pred_tail)
        metrics["Tail_MAE"] = compute_mae(y_true_tail, y_pred_tail)

    else:
        metrics["Tail_HR"] = 0.0
        metrics["Tail_MAE"] = 0.0


    return metrics
