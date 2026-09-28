
"""Run the manuscript protocol: stratified five-fold CV repeated with five seeds."""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from fold_preprocessing import prepare_index_split
from data_preprocessing import create_dataloaders, load_raw_data
from graph_construction import build_full_graph
from models.gad_net import GADNet
from train import evaluate_model, train_gad_net
from evaluate import compute_all_metrics

SEEDS = (48, 60, 72, 66, 67)


def set_run_seed(seed):
    """Set all available RNGs for one matched fold-seed evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(num_features):
    return GADNet(
        num_features=num_features,
        hidden_dim=config.HIDDEN_DIM,
        num_gnn_layers=config.NUM_GNN_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        num_edge_types=config.NUM_EDGE_TYPES,
        dropout=config.DROPOUT,
        node_embed_dim=config.NODE_EMBED_DIM,
        use_graph_learning=True,
        use_attention=True,
        use_hetero=True,
        use_knowledge_gate=True,
        use_node_embed=True,
        use_multi_level_fusion=True,
    )


# Stratify by yield-strength deciles and evaluate each test fold once.
def split_plan(y, seed):
    labels = pd.qcut(y, q=10, labels=False, duplicates="drop").astype(int)
    indices = np.arange(len(y))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_val_pos, test_pos) in enumerate(splitter.split(indices, labels), 1):
        train_val = indices[train_val_pos]
        test = indices[test_pos]
        train, val = train_test_split(
            train_val,
            test_size=15 / 85,
            random_state=seed + fold,
            stratify=labels[train_val],
        )
        yield fold, train, val, test


def aggregate(rows):
    if not rows:
        raise ValueError("No completed runs to aggregate")
    result = {}
    for metric in sorted(rows[0]["metrics"]):
        values = np.asarray([row["metrics"][metric] for row in rows], dtype=float)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        result[metric] = {
            "mean": mean,
            "std": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "n": int(len(values)),
        }
    return result

# Run the repeated five-fold protocol.
def run_main(data_path, device, output):
    os.makedirs(output, exist_ok=True)
    target = load_raw_data(data_path)[config.TARGET_COL].to_numpy(dtype=float)
    rows = []
    for seed in SEEDS:
        for fold, train_idx, val_idx, test_idx in split_plan(target, seed):
            run_id = f"seed{seed}_fold{fold}"
            set_run_seed(seed * 100 + fold)
            data = prepare_index_split(data_path, train_idx, val_idx, test_idx)
            graph = build_full_graph(data)
            train_loader, val_loader, test_loader = create_dataloaders(data)
            model = make_model(data["num_features"])
            previous_device = config.DEVICE
            config.DEVICE = device
            try:
                model, _ = train_gad_net(
                    model, train_loader, val_loader, graph, data,
                    use_adaboost=True, verbose=False)
                predicted, observed, _ = evaluate_model(
                    model, test_loader, data["scaler_y"], graph_data=graph,
                    model_type="gad_net", tail_bounds=data["tail_bounds"])
            finally:
                config.DEVICE = previous_device
            metrics = compute_all_metrics(observed, predicted,
                                          tail_bounds=data["tail_bounds"])
            row = {
                "seed": int(seed), "fold": int(fold), "run_id": run_id,
                "train_n": int(len(train_idx)), "val_n": int(len(val_idx)),
                "test_n": int(len(test_idx)),
                "metrics": {key: float(value) for key, value in metrics.items()},
                "train_indices": train_idx.tolist(),
                "val_indices": val_idx.tolist(),
                "test_indices": test_idx.tolist(),
            }
            rows.append(row)
            with open(os.path.join(output, run_id + ".json"), "w", encoding="utf-8") as handle:
                json.dump(row, handle, ensure_ascii=False)
            print("[main]", run_id, metrics["MAE"], metrics["RMSE"], metrics["R2"], flush=True)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU")
    parser.add_argument("--output", default="results_5fold_cv/main")
    args = parser.parse_args()
    device = (torch.device(f"cuda:{args.gpu}")
              if args.gpu >= 0 and torch.cuda.is_available()
              else torch.device("cpu"))
    rows = run_main(args.data, device, args.output)
    payload = {
        "metadata": {
            "protocol": "stratified five-fold CV repeated with five independent random seeds; 20% test fold; validation is 15% of the remaining data",
            "seeds": list(SEEDS),
            "gpu": args.gpu,
            "data": args.data,
            "stratification": "yield-strength deciles",
            "preprocessing_fit_on": "training fold only",
            "test_evaluated_once": True,
            "model": "GAD-Net main model",
        },
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    with open(os.path.join(args.output, "main_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
