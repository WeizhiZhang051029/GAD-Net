"""
Data Preprocessing Module for CAPL Steel Quality Prediction
Handles: loading (with CN->EN renaming), cleaning, normalization, splitting
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")
import config


# Apply fixed physical bounds before creating data splits.
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
    mi_scores = mutual_info_regression(X, y)
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
