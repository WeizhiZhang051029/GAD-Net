"""Knowledge-guided adaptive graph network for steel quality prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import config
from graph_construction import AdaptiveGraphLearningLayer


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for feature-level attention."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, N, D) - batch of node features
        Returns:
            out: (B, N, D) attended features
            attn_weights: (B, H, N, N) attention weights
        """
        B, N, D = x.shape
        Q = self.W_Q(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = attn_weights @ V
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(context)
        return out, attn_weights


# Relational graph convolution with basis decomposition.
class RGCNLayer(nn.Module):
    """
    Relational Graph Convolutional Network Layer.
    Handles heterogeneous edge types via basis decomposition.
    Following Eq. 12-13 in reference paper.
    """

    def __init__(self, in_dim, out_dim, num_edge_types, num_bases=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_edge_types = num_edge_types
        self.num_bases = min(num_bases, num_edge_types)


        self.bases = nn.Parameter(
            torch.Tensor(self.num_bases, in_dim, out_dim)
        )
        self.coefficients = nn.Parameter(
            torch.Tensor(num_edge_types, self.num_bases)
        )

        self.W_self = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(self, x, adj_tensor):
        """
        Args:
            x: (B, N, D_in) node features
            adj_tensor: (K, N, N) heterogeneous adjacency tensor
        Returns:
            out: (B, N, D_out) updated node features
        """
        B, N, D = x.shape


        W_r = torch.einsum('kb,bdi->kdi', self.coefficients, self.bases)


        out = torch.zeros(B, N, self.out_dim, device=x.device)

        for k in range(self.num_edge_types):
            A_k = adj_tensor[k]
            if A_k.abs().sum() < 1e-8:
                continue


            deg = A_k.abs().sum(dim=0, keepdim=True).clamp(min=1e-6)
            A_k_norm = A_k / deg


            neighbor_msg = torch.matmul(A_k_norm.T, x)
            neighbor_msg = torch.matmul(neighbor_msg, W_r[k])
            out = out + neighbor_msg


        out = out + self.W_self(x)
        out = out + self.bias
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GNNAttentionBlock(nn.Module):
    """
    Single GNN-Attention block combining:
    1. Multi-head self-attention (intra-node temporal-like attention)
    2. RGCN for spatial (inter-node) aggregation
    3. FFN for deep feature interaction
    4. Residual connections + LayerNorm
    """

    def __init__(self, hidden_dim, num_edge_types, num_heads=4,
                 num_bases=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.rgcn = RGCNLayer(hidden_dim, hidden_dim, num_edge_types,
                              num_bases, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_tensor):
        """
        Args:
            x: (B, N, D) node features
            adj_tensor: (K, N, N) heterogeneous adjacency tensor
        Returns:
            out: (B, N, D) updated features
            attn_weights: (B, H, N, N) attention weights
        """

        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)


        rgcn_out = self.rgcn(x, adj_tensor)
        x = self.norm2(x + rgcn_out)


        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x, attn_weights


# Main GAD-Net architecture.
class GADNet(nn.Module):
    """GAD-Net architecture with prior-guided graph learning and attention."""

    def __init__(self, num_features, hidden_dim=64, num_gnn_layers=3,
                 num_attention_heads=4, num_edge_types=16, dropout=0.2,
                 node_embed_dim=32, use_graph_learning=True,
                 use_attention=True, use_hetero=True,
                 use_knowledge_gate=True, use_node_embed=True,
                 use_multi_level_fusion=True):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_edge_types = num_edge_types if use_hetero else 1
        self.use_graph_learning = use_graph_learning
        self.use_attention = use_attention
        self.use_hetero = use_hetero
        self.use_knowledge_gate = use_knowledge_gate
        self.use_node_embed = use_node_embed
        self.use_multi_level_fusion = use_multi_level_fusion


        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


        if use_graph_learning:
            self.graph_learning = AdaptiveGraphLearningLayer(
                num_nodes=num_features,
                node_embed_dim=node_embed_dim,
                num_edge_types=self.num_edge_types,
                knowledge_gate_beta=config.KNOWLEDGE_GATE_BETA if use_knowledge_gate else 0.0,
                gate_slope_mu=config.GATE_SLOPE_MU,
                use_node_embed=use_node_embed,
            )


        if use_attention:
            self.gnn_blocks = nn.ModuleList([
                GNNAttentionBlock(
                    hidden_dim, self.num_edge_types,
                    num_heads=num_attention_heads,
                    num_bases=min(4, self.num_edge_types),
                    dropout=dropout,
                )
                for _ in range(num_gnn_layers)
            ])
        else:

            self.gnn_blocks = nn.ModuleList([
                nn.ModuleDict({
                    'rgcn': RGCNLayer(hidden_dim, hidden_dim,
                                      self.num_edge_types, dropout=dropout),
                    'ffn': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    'norm': nn.LayerNorm(hidden_dim),
                })
                for _ in range(num_gnn_layers)
            ])


        if use_multi_level_fusion:
            self.layer_weights = nn.Parameter(
                torch.ones(num_gnn_layers) / num_gnn_layers
            )


        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * num_features, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, node_types, adj_mechanism=None,
                adj_het_structure=None, adj_fixed=None):
        """
        Forward pass.

        Args:
            x: (B, N) input features (each sample has N feature values)
            node_types: (N,) node type indices
            adj_mechanism: (N, N) mechanism knowledge graph
            adj_het_structure: (K, N, N) heterogeneous structure template
            adj_fixed: (K, N, N) fixed adjacency (when graph learning is off)

        Returns:
            pred: (B, 1) predictions
            learned_adj: (K, N, N) learned adjacency tensor
            attn_weights_list: list of attention weight tensors
        """
        B, N = x.shape


        h = x.unsqueeze(-1)
        h = self.input_proj(h)


        # Fuse the mechanism prior with the learned topology.
        if self.use_graph_learning:

            mech = adj_mechanism if self.use_knowledge_gate else None
            het = adj_het_structure if self.use_hetero else None
            learned_adj, a_G = self.graph_learning(node_types, mech, het)
        else:


            K = self.num_edge_types

            if adj_het_structure is not None and adj_het_structure.dim() == 3:

                learned_adj = adj_het_structure
            elif adj_fixed is not None:
                if adj_fixed.dim() == 3:

                    learned_adj = adj_fixed
                else:

                    learned_adj = adj_fixed.unsqueeze(0).expand(K, -1, -1)
            elif adj_mechanism is not None:

                learned_adj = adj_mechanism.unsqueeze(0).expand(K, -1, -1)
            else:

                learned_adj = torch.ones(K, N, N, device=x.device)

            a_G = None


        layer_outputs = []
        attn_weights_list = []

        for l, block in enumerate(self.gnn_blocks):
            if self.use_attention:
                h, attn_w = block(h, learned_adj)
                attn_weights_list.append(attn_w)
            else:
                h_rgcn = block['rgcn'](h, learned_adj)
                h_ffn = block['ffn'](h_rgcn)
                h = block['norm'](h + h_ffn)
            layer_outputs.append(h)


        if self.use_multi_level_fusion and len(layer_outputs) > 1:
            weights = F.softmax(self.layer_weights, dim=0)
            h_fused = sum(w * out for w, out in zip(weights, layer_outputs))
        else:
            h_fused = layer_outputs[-1]


        h_flat = h_fused.reshape(B, -1)
        pred = self.output_head(h_flat)

        return pred, learned_adj, attn_weights_list

    def get_graph_regularization(self, learned_adj):
        """
        Differentiable sparsity regularization for learned heterogeneous adjacency.
        Normalize by the actual number of entries to avoid over-penalizing multi-relational graphs.
        """
        if learned_adj is not None and learned_adj.requires_grad:
            return learned_adj.abs().sum() / learned_adj.numel()

        return torch.tensor(
            0.0,
            device=learned_adj.device if learned_adj is not None else config.DEVICE
        )


# Tail-aware adaptive sample weighting.
class AdaBoostWeightManager:
    """
    Error-driven tail-aware dynamic reweighting (stable EMA version).

    Core idea is preserved:
        final_weight = base_weights × multiplier

    - base_weights = original sw_train (never destroyed)
    - body multiplier = 1.0 (frozen)
    - tail multiplier = dynamically updated every epoch
    - update signal = EMA-smoothed prediction error
    - normalization = robust percentile-based scaling
    """

    def __init__(
            self,
            num_samples,
            base_weights,
            y_train,
            boost_lr=0.024,
            tail_boost=1.20,
            tail_percentile=10,
            ema_beta=0.96,
            robust_q=80.0,
            min_tail_multiplier=1.00,
            max_tail_multiplier=1.45,
            update_momentum=0.90,
            mean_anchor=0.04,
    ):
        self.num_samples = int(num_samples)
        self.boost_lr = float(boost_lr)
        self.tail_boost = float(tail_boost)
        self.ema_beta = float(ema_beta)
        self.robust_q = float(robust_q)
        self.min_tail_multiplier = float(min_tail_multiplier)
        self.max_tail_multiplier = float(max_tail_multiplier)
        self.update_momentum = float(update_momentum)
        self.mean_anchor = float(mean_anchor)


        self.base_weights = np.asarray(base_weights, dtype=np.float64).copy()


        y = y_train.ravel() if hasattr(y_train, "ravel") else np.asarray(y_train)
        lo = np.percentile(y, tail_percentile)
        hi = np.percentile(y, 100 - tail_percentile)
        self.is_tail = (y <= lo) | (y >= hi)

        n_tail = int(self.is_tail.sum())
        n_body = int(self.num_samples - n_tail)


        self.multipliers = np.ones(self.num_samples, dtype=np.float64)
        self.multipliers[self.is_tail] = self.tail_boost


        self.ema_errors = np.zeros(self.num_samples, dtype=np.float64)


        self.multiplier_history = [self.multipliers.copy()]
        self.alpha_history = []
        self.error_history = []
        self.current_round = 0

        print(f"  [TailAware EMA] {n_tail} tail + {n_body} body samples")
        print(
            f"  [TailAware EMA] tail_boost={self.tail_boost}, "
            f"boost_lr={self.boost_lr}, ema_beta={self.ema_beta}, robust_q={self.robust_q}"
        )
        print("  [TailAware EMA] Body multiplier: FROZEN at 1.0")
        print(
            f"  [TailAware EMA] Tail multiplier range: "
            f"[{self.min_tail_multiplier:.2f}, {self.max_tail_multiplier:.2f}]"
        )

    def update_weights(self, errors, epoch):
        """
        Update tail multipliers EVERY EPOCH with small, smooth steps.

        Steps:
        1) EMA smooth the per-sample absolute errors
        2) Use robust percentile scaling inside tail samples
        3) Center tail difficulty by median (not mean)
        4) Exponential small-step update
        5) Momentum-style smoothing on multipliers
        """
        errors = np.abs(np.asarray(errors, dtype=np.float64))
        if errors.shape[0] != self.num_samples:
            raise ValueError(
                f"errors length {errors.shape[0]} != num_samples {self.num_samples}"
            )


        self.ema_errors = self.ema_beta * self.ema_errors + (1.0 - self.ema_beta) * errors

        tail_errors = self.ema_errors[self.is_tail]
        if tail_errors.size == 0:
            return self.get_combined_weights()


        scale = np.percentile(tail_errors, self.robust_q)
        if (not np.isfinite(scale)) or scale < 1e-10:
            scale = np.mean(tail_errors)
        if (not np.isfinite(scale)) or scale < 1e-10:
            scale = np.max(tail_errors)
        if (not np.isfinite(scale)) or scale < 1e-10:
            return self.get_combined_weights()


        difficulty = np.clip(self.ema_errors / (scale + 1e-12), 0.0, 1.30)

        tail_difficulty = difficulty[self.is_tail]
        center = np.median(tail_difficulty)
        delta = tail_difficulty - center


        delta = np.clip(delta, -0.24, 0.24)

        target_tail = self.multipliers[self.is_tail] * np.exp(self.boost_lr * delta)


        target_tail = np.clip(
            target_tail,
            self.min_tail_multiplier,
            self.max_tail_multiplier
        )


        tail_mean = target_tail.mean()
        if tail_mean > 1e-12:
            anchored_mean = (1.0 - self.mean_anchor) * tail_mean + self.mean_anchor * self.tail_boost
            target_tail = target_tail * (anchored_mean / tail_mean)

        target_tail = np.clip(
            target_tail,
            self.min_tail_multiplier,
            self.max_tail_multiplier
        )


        current_tail = self.multipliers[self.is_tail]
        new_tail = (
            self.update_momentum * current_tail
            + (1.0 - self.update_momentum) * target_tail
        )

        new_tail = np.clip(
            new_tail,
            self.min_tail_multiplier,
            self.max_tail_multiplier
        )

        self.multipliers[self.is_tail] = new_tail
        self.multipliers[~self.is_tail] = 1.0

        self.multiplier_history.append(self.multipliers.copy())
        self.current_round += 1

        tail_m = self.multipliers[self.is_tail]
        if self.current_round <= 3 or epoch % 20 == 0:
            print(
                f"  [TailAware EMA] E{epoch:03d}/R{self.current_round}: "
                f"tail=[{tail_m.min():.3f},{tail_m.max():.3f}] "
                f"mean={tail_m.mean():.3f}, body=1.000"
            )

        return self.get_combined_weights()

    def get_combined_weights(self):
        """Return actual training weights = base_weights × multipliers."""
        return self.base_weights * self.multipliers

    def get_weights_tensor(self, indices=None):
        """Get combined weights as a PyTorch tensor."""
        cw = self.get_combined_weights()
        if indices is not None:
            return torch.FloatTensor(cw[indices])
        return torch.FloatTensor(cw)

    def get_weight_history(self):
        """Return multiplier history for visualization."""
        return np.array(self.multiplier_history)

    def get_boosting_stats(self):
        """Return alpha and error histories for visualization."""
        return {
            "alphas": np.array(self.alpha_history),
            "errors": np.array(self.error_history),
        }

    def get_tail_stats(self):
        """Convenient monitoring stats."""
        tail_m = self.multipliers[self.is_tail]
        return {
            "tail_min": float(tail_m.min()) if tail_m.size > 0 else 1.0,
            "tail_max": float(tail_m.max()) if tail_m.size > 0 else 1.0,
            "tail_mean": float(tail_m.mean()) if tail_m.size > 0 else 1.0,
        }


# Weighted objective with graph regularization.
class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with graph regularization.
    Loss = ξ₀ * Σ(w_i * (ŷ_i - y_i)²) + ξ₁ * L1 + ξ₂ * (e_R / e_full)
    Following Eq. 15 in reference paper.
    """

    def __init__(self, mse_weight=1.0, l1_weight=0.001, graph_weight=0.01):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.graph_weight = graph_weight

    def forward(self, pred, target, sample_weights=None,
                model=None, learned_adj=None):
        """
        Args:
            pred: (B, 1) predictions
            target: (B, 1) targets
            sample_weights: (B,) per-sample weights
            model: the model (for L1 regularization)
            learned_adj: (K, N, N) learned graph (for sparsity reg)
        """

        mse_per_sample = (pred - target).pow(2).squeeze(-1)
        if sample_weights is not None:
            sw = sample_weights / sample_weights.sum() * len(sample_weights)
            loss_mse = (sw * mse_per_sample).mean()
        else:
            loss_mse = mse_per_sample.mean()

        total_loss = self.mse_weight * loss_mse


        if model is not None and self.l1_weight > 0:
            l1_terms = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("graph_learning."):
                    continue
                l1_terms.append(p.abs().mean())
            if l1_terms:
                l1_reg = torch.stack(l1_terms).sum()
                total_loss = total_loss + self.l1_weight * l1_reg


        if learned_adj is not None and self.graph_weight > 0:
            graph_reg = model.get_graph_regularization(learned_adj)
            total_loss = total_loss + self.graph_weight * graph_reg

        return total_loss, loss_mse


if __name__ == "__main__":

    B, N, K = 16, 22, 16
    model = GADNet(num_features=N, hidden_dim=64, num_gnn_layers=3,
                   num_attention_heads=4, num_edge_types=K)
    x = torch.randn(B, N)
    node_types = torch.randint(0, 4, (N,))
    adj_mech = torch.rand(N, N)
    adj_het = torch.rand(K, N, N)
    pred, adj, attn_list = model(x, node_types, adj_mech, adj_het)
    print(f"Pred shape: {pred.shape}")
    print(f"Learned adj shape: {adj.shape}")
    print(f"Num attention layers: {len(attn_list)}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
