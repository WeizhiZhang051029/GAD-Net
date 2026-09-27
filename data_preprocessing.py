"""
Data Preprocessing Module for CAPL Steel Quality Prediction
Handles: loading (with CN->EN renaming), cleaning, normalization, splitting
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")
import config


def apply_physical_constraints(df):
    """Remove records outside fixed physical admissibility bounds."""
    mask = pd.Series(True, index=df.index)
    for column, (lower, upper) in config.PHYSICAL_BOUNDS.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if lower is not None:
            mask &= values >= lower
        if upper is not None:
            mask &= values <= upper
    removed = int((~mask).sum())
    if removed:
        print(f"[Data] Physical-constraint filtering: {len(df)} -> {len(df) - removed} ({removed} removed)")
    return df.loc[mask].reset_index(drop=True)


class CAPLDataset(Dataset):
    def __init__(self, X, y, sample_weights=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.sample_weights = torch.FloatTensor(sample_weights) \
            if sample_weights is not None else torch.ones(len(y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sample_weights[idx], idx


def load_raw_data(path=None):
    if path is None:
        path = config.DATA_PATH
    df = pd.read_csv(path, encoding="gb18030") if str(path).lower().endswith(".csv") else pd.read_excel(path)
    # Rename Chinese columns to English
    df = df.rename(columns=config.COLUMN_NAME_MAP)
    if df.isna().any().any():
        missing = df.columns[df.isna().any()].tolist()
        raise ValueError(f"Missing values found in columns: {missing}")
    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Non-finite numeric values found in the input data")
    df = apply_physical_constraints(df)
    print(f"[Data] Loaded {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[Data] Columns: {list(df.columns)}")
    return df



def compute_sample_imbalance_weights(y, num_bins=10, strategy="inverse_freq", bins=None):
    """Compute sample weights using bin edges fitted on the training labels."""
    if bins is None:
        bins = np.linspace(y.min() - 1e-6, y.max() + 1e-6, num_bins + 1)
    bin_indices = np.clip(np.digitize(y, bins) - 1, 0, num_bins - 1)
    bin_counts = np.maximum(np.bincount(bin_indices, minlength=num_bins).astype(float), 1)
    if strategy == "inverse_freq":
        bin_weights = 1.0 / bin_counts
    elif strategy == "sqrt_inverse_freq":
        bin_weights = 1.0 / np.sqrt(bin_counts)
    else:
        bin_weights = np.ones(num_bins)
    sample_weights = bin_weights[bin_indices]
    sample_weights = sample_weights / sample_weights.sum() * len(y)
    return sample_weights, bin_indices, bin_counts, bins


def compute_feature_correlations(X, y, feature_names):
    df_full = pd.DataFrame(X, columns=feature_names)
    df_full[config.TARGET_COL] = y
    corr_matrix = df_full[feature_names].corr().values
    mi_scores = mutual_info_regression(X, y, random_state=config.SEED)
    mi_max = float(np.max(mi_scores)) if mi_scores.size else 0.0
    if mi_max > 0.0 and np.isfinite(mi_max):
        mi_scores = mi_scores / mi_max
    else:
        mi_scores = np.zeros_like(mi_scores, dtype=float)
    target_corr = df_full.corr()[config.TARGET_COL].drop(config.TARGET_COL)
    feature_corr_df = pd.DataFrame({
        "feature": feature_names,
        "pearson_corr": target_corr.values,
        "abs_pearson": np.abs(target_corr.values),
        "mutual_info": mi_scores,
    }).sort_values("abs_pearson", ascending=False)
    return corr_matrix, mi_scores, feature_corr_df


def preprocess_data(data_path=None, return_raw=False, split_seed=None):
    """Load and split data with split-specific, train-only preprocessing.

    The split is stratified by yield-strength bins.  All fitted preprocessing
    objects (feature/target scalers and inverse-frequency weights) are fitted
    on the training partition and then applied to validation/test partitions.
    """
    df = load_raw_data(data_path)
    # Do not remove observations using statistics computed from the full data.
    # The revised evaluation keeps all records and fits preprocessing on train only.

    feature_names = [c for c in df.columns if c != config.TARGET_COL]
    X_raw = df[feature_names].values.astype(np.float64)
    y_raw = df[config.TARGET_COL].values.astype(np.float64)

    if split_seed is None:
        split_seed = config.SEED

    # Stratification labels are formed only to create the requested split.
    split_labels = pd.qcut(y_raw, q=10, labels=False, duplicates="drop").astype(int)
    indices = np.arange(len(y_raw))
    idx_tv, idx_test = train_test_split(
        indices, test_size=config.TEST_RATIO, random_state=split_seed,
        stratify=split_labels)
    val_adj = config.VAL_RATIO / (1 - config.TEST_RATIO)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=val_adj, random_state=split_seed,
        stratify=split_labels[idx_tv])

    X_train_raw, X_val_raw, X_test_raw = X_raw[idx_train], X_raw[idx_val], X_raw[idx_test]
    y_train_raw, y_val_raw, y_test_raw = y_raw[idx_train], y_raw[idx_val], y_raw[idx_test]

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_test = scaler_X.transform(X_test_raw)
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # Fit bin edges and inverse-frequency weights on training labels only.
    sw_train, bi_train, bin_counts, weight_bins = compute_sample_imbalance_weights(
        y_train_raw, num_bins=10, strategy="inverse_freq")
    _, bi_val, _, _ = compute_sample_imbalance_weights(
        y_val_raw, num_bins=10, strategy="inverse_freq", bins=weight_bins)
    _, bi_test, _, _ = compute_sample_imbalance_weights(
        y_test_raw, num_bins=10, strategy="inverse_freq", bins=weight_bins)
    sw_val = np.ones(len(y_val), dtype=np.float64)
    sw_test = np.ones(len(y_test), dtype=np.float64)

    # Correlation/MI graph statistics are also estimated from training data only.
    corr_matrix, mi_scores, feature_corr_df = compute_feature_correlations(
        X_train_raw, y_train_raw, feature_names)

    print(f"[Data] Split: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

    node_types = np.array([config.NODE_TYPE_MAP[n] for n in feature_names])

    data_dict = {
        "feature_names": feature_names, "num_features": len(feature_names),
        "node_types": node_types,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "sw_train": sw_train, "sw_val": sw_val, "sw_test": sw_test,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "corr_matrix": corr_matrix, "mi_scores": mi_scores,
        "feature_corr_df": feature_corr_df, "bin_counts": bin_counts,
        "y_raw_train": y_train_raw,
        "y_raw_val": y_val_raw,
        "y_raw_test": y_test_raw,
        "split_seed": int(split_seed),
        "train_indices": idx_train,
        "val_indices": idx_val,
        "test_indices": idx_test,
        "weight_bins": weight_bins,
    }
    if return_raw:
        data_dict["X_raw"] = X_raw
        data_dict["y_raw"] = y_raw
    return data_dict


def create_dataloaders(data_dict, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    train_ds = CAPLDataset(data_dict["X_train"], data_dict["y_train"], data_dict["sw_train"])
    val_ds   = CAPLDataset(data_dict["X_val"], data_dict["y_val"], data_dict["sw_val"])
    test_ds  = CAPLDataset(data_dict["X_test"], data_dict["y_test"], data_dict["sw_test"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
