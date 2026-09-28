"""
Training Pipeline for GAD-Net
Includes: training loop, AdaBoost weight updates, early stopping, logging
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

import config
from models.gad_net import GADNet, AdaBoostWeightManager, WeightedMSELoss
from evaluate import compute_all_metrics
from graph_construction import build_heterogeneous_adjacency_tensor


class EarlyStopping:
    """Early stopping with patience."""
    def __init__(self, patience=30, min_delta=1e-6, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def step(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = (score < self.best_score - self.min_delta) if self.mode == "min" \
            else (score > self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# Optimize on the training fold and select by validation loss.
def train_gad_net(model, train_loader, val_loader, graph_data, data_dict,
                  num_epochs=None, use_adaboost=True, verbose=True,
                  weight_vis_callback=None):
    """
    Train GAD-Net with dynamic tail-aware adaptive sample weighting.

    Args:
        model: GADNet instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        graph_data: graph structure data
        data_dict: preprocessed data dictionary
        num_epochs: number of training epochs
        use_adaboost: whether to use adaptive sample weighting
        verbose: print training progress
        weight_vis_callback: callback for weight visualization
    Returns:
        model, history
    """
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS

    device = config.DEVICE
    model = model.to(device)


    node_types = graph_data["node_types"].to(device)
    adj_mechanism = graph_data["adj_mechanism"].to(device)
    adj_het = graph_data["adj_het_tensor"].to(device)
    # Relation-aligned mechanistic prior used by Eq. (11).
    if model.use_hetero:
        prior_adj = torch.as_tensor(build_heterogeneous_adjacency_tensor(
            graph_data["adj_mechanism"].cpu().numpy(),
            graph_data["node_types"].cpu().numpy()), device=device)
    else:
        prior_adj = adj_mechanism.unsqueeze(0)


    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_DECAY_STEP,
        gamma=config.LR_DECAY_GAMMA
    )


    criterion = WeightedMSELoss(
        mse_weight=config.LOSS_MSE_WEIGHT,
        l1_weight=config.LOSS_L1_WEIGHT,
        graph_weight=config.LOSS_GRAPH_WEIGHT,
    )


    n_train = len(train_loader.dataset)
    adaboost_mgr = None
    if use_adaboost:
        sw_train = data_dict["sw_train"]
        y_train = data_dict["y_train"]
        adaboost_mgr = AdaBoostWeightManager(
            n_train,
            base_weights=sw_train,
            y_train=y_train,
            boost_lr=config.ADABOOST_LEARNING_RATE,
            tail_boost=config.ADABOOST_TAIL_BOOST,
            tail_percentile=config.ADABOOST_TAIL_PERCENTILE,
            ema_beta=config.ADABOOST_ERROR_EMA,
            robust_q=config.ADABOOST_ROBUST_QUANTILE,
            min_tail_multiplier=config.ADABOOST_MIN_TAIL_MULTIPLIER,
            max_tail_multiplier=config.ADABOOST_MAX_TAIL_MULTIPLIER,
            update_momentum=config.ADABOOST_UPDATE_MOMENTUM,
        )


    early_stop = EarlyStopping(
        patience=config.EARLY_STOP_PATIENCE,
        mode="min"
    )


    history = defaultdict(list)
    best_model_state = None

    for epoch in range(1, num_epochs + 1):


        # Training phase.
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []
        all_train_indices = []

        for batch_X, batch_y, batch_sw, batch_idx in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_idx_np = batch_idx.detach().cpu().numpy()


            if adaboost_mgr is not None:
                sw = adaboost_mgr.get_weights_tensor(batch_idx_np).to(device)
            else:
                sw = torch.ones_like(batch_sw, device=device)

            optimizer.zero_grad()

            pred, learned_adj, attn_list = model(
                batch_X, node_types, adj_mechanism, adj_het
            )

            loss, mse_loss = criterion(pred, batch_y, sw, model, learned_adj, prior_adj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())
            all_train_preds.append(pred.detach().cpu().numpy())
            all_train_targets.append(batch_y.detach().cpu().numpy())
            all_train_indices.append(batch_idx_np)


        if adaboost_mgr is not None and len(all_train_preds) > 0:
            preds_arr = np.concatenate(all_train_preds).ravel()
            targets_arr = np.concatenate(all_train_targets).ravel()
            indices_arr = np.concatenate(all_train_indices)

            errors = np.zeros(n_train, dtype=np.float64)
            errors[indices_arr] = np.abs(preds_arr - targets_arr)

            adaboost_mgr.update_weights(errors, epoch)

        scheduler.step()


        if weight_vis_callback and epoch % config.WEIGHT_VIS_INTERVAL == 0:
            weight_vis_callback(epoch, adaboost_mgr)


        # Validation phase.
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_X, batch_y, batch_sw, batch_idx in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                pred, _, _ = model(batch_X, node_types, adj_mechanism, adj_het)
                mse = (pred - batch_y).pow(2).mean()

                val_losses.append(mse.item())
                val_preds.append(pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])


        val_preds_arr = np.concatenate(val_preds).reshape(-1, 1)
        val_targets_arr = np.concatenate(val_targets).reshape(-1, 1)

        scaler_y = data_dict["scaler_y"]
        val_preds_raw = scaler_y.inverse_transform(val_preds_arr).ravel()
        val_targets_raw = scaler_y.inverse_transform(val_targets_arr).ravel()

        val_metrics = compute_all_metrics(val_targets_raw, val_preds_raw)
        val_rmse = float(val_metrics["RMSE"])
        val_mape = float(val_metrics["MAPE"])
        val_r2 = float(val_metrics["R2"])

        if "y_raw_train" in data_dict:
            y_train_raw = np.asarray(data_dict["y_raw_train"]).ravel()
        else:
            y_train_raw = scaler_y.inverse_transform(
                data_dict["y_train"].reshape(-1, 1)
            ).ravel()

        lo = np.percentile(y_train_raw, config.ADABOOST_TAIL_PERCENTILE)
        hi = np.percentile(y_train_raw, 100 - config.ADABOOST_TAIL_PERCENTILE)
        val_tail_mask = (val_targets_raw <= lo) | (val_targets_raw >= hi)

        if np.any(val_tail_mask):
            val_tail_mae = float(
                np.mean(np.abs(val_preds_raw[val_tail_mask] - val_targets_raw[val_tail_mask]))
            )
        else:
            val_tail_mae = float(np.mean(np.abs(val_preds_raw - val_targets_raw)))


        selection_score = avg_val_loss

        history["val_rmse"].append(val_rmse)
        history["val_mape"].append(val_mape)
        history["val_r2"].append(val_r2)
        history["val_tail_mae"].append(val_tail_mae)
        history["selection_score"].append(float(selection_score))


        if adaboost_mgr is not None:
            if "boosting_val_rmse" not in history:
                history["boosting_val_rmse"] = []
            history["boosting_val_rmse"].append(val_rmse)

            if hasattr(adaboost_mgr, "get_tail_stats"):
                tail_stats = adaboost_mgr.get_tail_stats()
                history["tail_min_multiplier"].append(tail_stats["tail_min"])
                history["tail_max_multiplier"].append(tail_stats["tail_max"])
                history["tail_mean_multiplier"].append(tail_stats["tail_mean"])


        if early_stop.best_score is None or selection_score < early_stop.best_score:
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}


        if early_stop.step(selection_score, epoch):
            if verbose:
                print(
                    f"  [Early Stop] at epoch {epoch}, "
                    f"best selection_score={early_stop.best_score:.6f} "
                    f"at epoch {early_stop.best_epoch}"
                )
            break

        if verbose and epoch % 20 == 0:
            print(
                f"  Epoch {epoch:3d}/{num_epochs} | "
                f"RMSE: {val_rmse:.4f} | "
                f"MAPE: {val_mape:.4f} | "
                f"R2: {val_r2:.4f} | "
                f"Tail_MAE: {val_tail_mae:.4f} | "
                f"ValLoss: {selection_score:.6f} | "

                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )


    # Restore the checkpoint with the best validation loss.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)


    if adaboost_mgr is not None:
        history["adaboost_weight_history"] = adaboost_mgr.get_weight_history()
        history["adaboost_boosting_stats"] = adaboost_mgr.get_boosting_stats()


    model.eval()
    with torch.no_grad():
        probe_x = torch.zeros(1, model.num_features).to(device)
        _, final_adj, _ = model(probe_x, node_types, adj_mechanism, adj_het)
        history["learned_adj"] = final_adj.cpu().numpy()

        attn_accum = None
        n_batches = 0
        for batch_X, batch_y, batch_sw, batch_idx in train_loader:
            batch_X = batch_X.to(device)
            _, _, batch_attn = model(batch_X, node_types, adj_mechanism, adj_het)

            if batch_attn:
                if attn_accum is None:
                    attn_accum = [
                        a.detach().cpu().sum(dim=0).numpy()
                        for a in batch_attn
                    ]
                else:
                    for li, a in enumerate(batch_attn):
                        attn_accum[li] += a.detach().cpu().sum(dim=0).numpy()
                n_batches += batch_X.size(0)

        if attn_accum is not None and n_batches > 0:
            history["final_attn_weights"] = [acc / n_batches for acc in attn_accum]

    return model, history


def evaluate_model(model, test_loader, scaler_y, graph_data=None,
                   model_type="gad_net", adj=None, tail_bounds=None):
    """
    Evaluate model on test set, returning predictions and metrics.
    """
    device = config.DEVICE
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_X, batch_y, _, _ in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            if model_type == "gad_net" and graph_data is not None:
                node_types = graph_data["node_types"].to(device)
                adj_mech = graph_data["adj_mechanism"].to(device)
                adj_het = graph_data["adj_het_tensor"].to(device)
                pred, _, _ = model(batch_X, node_types, adj_mech, adj_het)
            else:
                if adj is not None:
                    adj_d = adj.to(device)
                    pred, _, _ = model(batch_X, adj=adj_d)
                else:
                    pred, _, _ = model(batch_X)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()


    preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    targets_orig = scaler_y.inverse_transform(targets.reshape(-1, 1)).ravel()

    metrics = compute_all_metrics(targets_orig, preds_orig, tail_bounds=tail_bounds)
    return preds_orig, targets_orig, metrics

