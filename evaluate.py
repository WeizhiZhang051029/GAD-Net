"""
Evaluation Module
Computes all metrics: MAE, MSE, RMSE, MAPE, R2, CORR, HR
"""
import numpy as np
from scipy import stats
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
    r, _ = stats.pearsonr(y_true, y_pred)
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


def format_metrics(metrics, prefix=""):
    """Format metrics dict as a readable string."""
    parts = []

    keys_order = ["MAE", "RMSE", "R2", "HR", "Tail_HR", "Tail_MAE", "MAPE", "CORR"]

    for k in keys_order:
        if k in metrics:
            v = metrics[k]
            if "HR" in k or "MAPE" in k:
                parts.append(f"{prefix}{k}: {v:.2f}%")
            elif k in ["R2", "CORR"]:
                parts.append(f"{prefix}{k}: {v:.4f}")
            else:
                parts.append(f"{prefix}{k}: {v:.4f}")


    for k, v in metrics.items():
        if k not in keys_order:
            parts.append(f"{prefix}{k}: {v:.4f}")

    return " | ".join(parts)


def aggregate_run_metrics(all_metrics_list):
    """
    Aggregate metrics from multiple independent runs.

    Args:
        all_metrics_list: list of metric dicts from each run
    Returns:
        agg: dict of metric_name -> {"mean": ..., "std": ...}
    """
    agg = {}
    keys = all_metrics_list[0].keys()
    for k in keys:
        vals = np.asarray([m[k] for m in all_metrics_list], dtype=float)
        n = len(vals)
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
        agg[k] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals.tolist(),
        }
    return agg


def paired_wilcoxon_holm(reference_rows, comparison_rows,
                         metric_names=("RMSE", "MAPE", "R2", "Tail_MAE")):
    """Run paired two-sided Wilcoxon tests and Holm correction."""
    def key(row):
        return row["run_id"]

    ref = {key(row): row for row in reference_rows}
    cmp = {key(row): row for row in comparison_rows}
    keys = sorted(set(ref).intersection(cmp))
    raw = []
    for metric in metric_names:
        x = np.asarray([ref[k]["metrics"][metric] for k in keys], dtype=float)
        y = np.asarray([cmp[k]["metrics"][metric] for k in keys], dtype=float)
        if len(x) < 2 or np.allclose(x, y):
            stat, pvalue = 0.0, 1.0
        else:
            try:
                result = stats.wilcoxon(x, y, alternative="two-sided",
                                        zero_method="wilcox", method="auto")
                stat, pvalue = float(result.statistic), float(result.pvalue)
            except ValueError:
                stat, pvalue = 0.0, 1.0
        raw.append({"metric": metric, "n_pairs": len(keys),
                    "statistic": stat, "p_raw": pvalue})
    order = sorted(range(len(raw)), key=lambda i: raw[i]["p_raw"])
    m = len(raw)
    adjusted = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * raw[i]["p_raw"])
        running = max(running, value)
        adjusted[i] = running
    for i, row in enumerate(raw):
        row["p_holm"] = float(adjusted[i])
        row["significant"] = bool(adjusted[i] < 0.05)
    return {"method": "two-sided paired Wilcoxon signed-rank; Holm correction",
            "pair_key": "run_id", "comparisons": raw}
