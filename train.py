"""
Training Pipeline for GAD-Net
Includes: training loop, AdaBoost weight updates, early stopping, logging
"""
import numpy as np
import torch
import torch.optim as optim

import config
from models.gad_net import AdaBoostWeightManager, WeightedMSELoss
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
                  num_epochs=None, use_adaboost=True, verbose=True):
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
    Returns:
        trained model
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


    best_model_state = None

    for epoch in range(1, num_epochs + 1):


        # Training phase.
        model.train()
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

            pred, learned_adj = model(batch_X, node_types, adj_mechanism, adj_het)

            loss, _ = criterion(pred, batch_y, sw, model, learned_adj, prior_adj)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

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



        # Validation phase.
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y, _, _ in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                pred, _ = model(batch_X, node_types, adj_mechanism, adj_het)
                mse = (pred - batch_y).pow(2).mean()

                val_losses.append(mse.item())

        avg_val_loss = np.mean(val_losses)

        selection_score = avg_val_loss


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
                f"ValLoss: {selection_score:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )


    # Restore the checkpoint with the best validation loss.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)


    return model




def evaluate_model(model, test_loader, scaler_y, graph_data, tail_bounds=None):
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

            node_types = graph_data["node_types"].to(device)
            adj_mech = graph_data["adj_mechanism"].to(device)
            adj_het = graph_data["adj_het_tensor"].to(device)
            pred, _ = model(batch_X, node_types, adj_mech, adj_het)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    preds = np.concatenate(all_preds).ravel()
    targets = np.concatenate(all_targets).ravel()


    preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    targets_orig = scaler_y.inverse_transform(targets.reshape(-1, 1)).ravel()

    metrics = compute_all_metrics(targets_orig, preds_orig, tail_bounds=tail_bounds)
    return preds_orig, targets_orig, metrics

