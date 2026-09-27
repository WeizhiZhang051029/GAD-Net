"""Leakage-safe preprocessing for stratified K-fold experiments."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_preprocessing import (
    load_raw_data, compute_sample_imbalance_weights,
    compute_feature_correlations,
)
import config


def prepare_index_split(data_path, train_idx, val_idx, test_idx, split_seed, return_raw=False):
    """Fit every preprocessing object on train_idx and apply it to val/test."""
    df = load_raw_data(data_path)
    feature_names = [c for c in df.columns if c != config.TARGET_COL]
    X_raw = df[feature_names].to_numpy(dtype=np.float64)
    y_raw = df[config.TARGET_COL].to_numpy(dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    Xtr_raw, Xv_raw, Xte_raw = X_raw[train_idx], X_raw[val_idx], X_raw[test_idx]
    ytr_raw, yv_raw, yte_raw = y_raw[train_idx], y_raw[val_idx], y_raw[test_idx]

    scaler_X = StandardScaler().fit(Xtr_raw)
    scaler_y = StandardScaler().fit(ytr_raw.reshape(-1, 1))
    Xtr, Xv, Xte = scaler_X.transform(Xtr_raw), scaler_X.transform(Xv_raw), scaler_X.transform(Xte_raw)
    ytr = scaler_y.transform(ytr_raw.reshape(-1, 1)).ravel()
    yv = scaler_y.transform(yv_raw.reshape(-1, 1)).ravel()
    yte = scaler_y.transform(yte_raw.reshape(-1, 1)).ravel()

    sw_train, _, bin_counts, weight_bins = compute_sample_imbalance_weights(ytr_raw, num_bins=10, strategy="inverse_freq")
    _, _, _, _ = compute_sample_imbalance_weights(yv_raw, num_bins=10, strategy="inverse_freq", bins=weight_bins)
    _, _, _, _ = compute_sample_imbalance_weights(yte_raw, num_bins=10, strategy="inverse_freq", bins=weight_bins)
    corr, mi, corr_df = compute_feature_correlations(Xtr_raw, ytr_raw, feature_names)
    node_types = np.array([config.NODE_TYPE_MAP[n] for n in feature_names], dtype=np.int64)
    out = {
        "feature_names": feature_names, "num_features": len(feature_names), "node_types": node_types,
        "X_train": Xtr, "y_train": ytr, "X_val": Xv, "y_val": yv, "X_test": Xte, "y_test": yte,
        "sw_train": sw_train, "sw_val": np.ones(len(yv)), "sw_test": np.ones(len(yte)),
        "scaler_X": scaler_X, "scaler_y": scaler_y, "corr_matrix": corr, "mi_scores": mi,
        "feature_corr_df": corr_df, "bin_counts": bin_counts, "weight_bins": weight_bins,
        "y_raw_train": ytr_raw, "y_raw_val": yv_raw, "y_raw_test": yte_raw,
        "split_seed": int(split_seed), "train_indices": train_idx, "val_indices": val_idx, "test_indices": test_idx,
        "tail_bounds": (float(np.percentile(ytr_raw, config.ADABOOST_TAIL_PERCENTILE)),
                        float(np.percentile(ytr_raw, 100 - config.ADABOOST_TAIL_PERCENTILE))),
    }
    if return_raw:
        out["X_raw"], out["y_raw"] = X_raw, y_raw
    return out
