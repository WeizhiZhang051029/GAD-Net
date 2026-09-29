"""
Graph Construction Module - English parameter names throughout
Builds heterogeneous graph from mechanism knowledge + data mining.
"""
import numpy as np
import torch
import torch.nn as nn
import config


# Encode process relations supplied by domain knowledge.
def build_mechanism_knowledge_graph(feature_names):
    """Build mechanism knowledge graph based on CAPL process domain knowledge."""
    N = len(feature_names)
    adj = np.zeros((N, N), dtype=np.float32)
    n2i = {name: i for i, name in enumerate(feature_names)}


    temp_chain = [
        "JPF_Preheat_Temp", "Heating_Furnace_Temp", "Soaking_Furnace_Temp",
        "Slow_Cooling_Temp", "Fast_Cooling1_Temp", "Overaging_Furnace_Temp",
        "Fast_Cooling2_Temp", "Quenching_Temp"
    ]
    for k in range(len(temp_chain) - 1):
        if temp_chain[k] in n2i and temp_chain[k+1] in n2i:
            adj[n2i[temp_chain[k]], n2i[temp_chain[k+1]]] = 1.0


    if "Furnace_Speed" in n2i:
        for t in temp_chain:
            if t in n2i:
                adj[n2i["Furnace_Speed"], n2i[t]] = 1.0


    for cp in ["C_Content", "Mn_Content", "S_Content", "P_Content"]:
        for mp in ["Elongation", "Rolling_Force", "Bending_Force"]:
            if cp in n2i and mp in n2i:
                adj[n2i[cp], n2i[mp]] = 1.0


    for hp in ["Heating_Temp", "Finish_Rolling_Temp", "Coiling_Temp"]:
        for cp in ["Cold_Rolling_Reduction", "Actual_Thickness"]:
            if hp in n2i and cp in n2i:
                adj[n2i[hp], n2i[cp]] = 1.0


    cools = ["Slow_Cooling_Temp", "Fast_Cooling1_Temp",
             "Overaging_Furnace_Temp", "Fast_Cooling2_Temp", "Quenching_Temp"]
    for ct in cools:
        for mp in ["Elongation", "Rolling_Force", "Bending_Force"]:
            if ct in n2i and mp in n2i:
                adj[n2i[ct], n2i[mp]] = 1.0


    if "Rolling_Force" in n2i:
        for p in ["Actual_Thickness", "Actual_Width", "Cold_Rolling_Reduction"]:
            if p in n2i:
                adj[n2i[p], n2i["Rolling_Force"]] = 1.0

    print(f"[Graph] Mechanism knowledge: {int((adj > 0).sum())} directed edges")
    return adj


def build_data_driven_graph(corr_matrix, mi_scores, feature_names,
                            corr_threshold=None):
    if corr_threshold is None:
        corr_threshold = config.CORRELATION_THRESHOLD
    N = len(feature_names)
    adj = np.zeros((N, N), dtype=np.float32)
    abs_corr = np.abs(corr_matrix)
    for i in range(N):
        for j in range(N):
            if i != j and abs_corr[i, j] >= corr_threshold:
                adj[i, j] = abs_corr[i, j]
    for i in range(N):
        adj[i, i] = max(mi_scores[i], 0.1)
    print(f"[Graph] Data-driven: {int((adj > 0).sum()) - N} edges")
    return adj


def build_heterogeneous_adjacency_tensor(adj_combined, node_types):
    N = len(node_types)
    K = config.NUM_EDGE_TYPES
    adj_tensor = np.zeros((K, N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if adj_combined[i, j] > 0:
                et = node_types[i] * config.NUM_NODE_TYPES + node_types[j]
                adj_tensor[et, i, j] = adj_combined[i, j]
    active = sum(1 for k in range(K) if (adj_tensor[k] > 0).any())
    print(f"[Graph] Heterogeneous tensor: {active} active edge types")
    return adj_tensor


# Merge the fixed prior with train-only data-driven relations.
def build_full_graph(data_dict):
    feature_names = data_dict["feature_names"]
    node_types = data_dict["node_types"]
    adj_mechanism = build_mechanism_knowledge_graph(feature_names)
    adj_data = build_data_driven_graph(
        data_dict["corr_matrix"], data_dict["mi_scores"], feature_names)
    adj_combined = np.maximum(adj_mechanism, adj_data)
    adj_het_tensor = build_heterogeneous_adjacency_tensor(adj_combined, node_types)
    return {
        "adj_mechanism": torch.FloatTensor(adj_mechanism),
        "adj_data": torch.FloatTensor(adj_data),
        "adj_combined": torch.FloatTensor(adj_combined),
        "adj_het_tensor": torch.FloatTensor(adj_het_tensor),
        "node_types": torch.LongTensor(node_types),
        "num_nodes": len(feature_names),
        "num_edge_types": config.NUM_EDGE_TYPES,
        "feature_names": feature_names,
    }


# Learn a globally shared soft topology gated by the mechanism prior.
class AdaptiveGraphLearningLayer(nn.Module):
    """Adaptive GL Layer with Knowledge-Gated Unit (Algorithm 1)."""
    def __init__(self, num_nodes, node_embed_dim, num_edge_types,
                 knowledge_gate_beta=1.0, gate_slope_mu=3.0,
                 use_node_embed=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.beta = knowledge_gate_beta
        self.mu = gate_slope_mu
        self.use_node_embed = use_node_embed
        self.node_embed_dim = node_embed_dim


        class_dim = node_embed_dim // 2
        indiv_dim = node_embed_dim - class_dim
        Fe = node_embed_dim

        if self.use_node_embed:

            self.node_class_embed = nn.Embedding(config.NUM_NODE_TYPES, class_dim)
            self.node_indiv_embed = nn.Embedding(num_nodes, indiv_dim)
            self.shared_node_token = None
        else:


            self.node_class_embed = None
            self.node_indiv_embed = None
            self.shared_node_token = nn.Parameter(torch.empty(1, node_embed_dim))
            nn.init.xavier_uniform_(self.shared_node_token)

        self.W_Q = nn.Linear(Fe, Fe, bias=False)
        self.W_K = nn.Linear(Fe, Fe, bias=False)
        self.W_G = nn.Linear(Fe, num_nodes, bias=False)

    def get_node_embeddings(self, node_types):
        if self.use_node_embed:
            class_emb = self.node_class_embed(node_types)
            indiv_emb = self.node_indiv_embed(
                torch.arange(self.num_nodes, device=node_types.device)
            )
            E = torch.cat([class_emb, indiv_emb], dim=-1)
        else:

            E = self.shared_node_token.expand(self.num_nodes, -1)
        return E

    def modified_sigmoid(self, x):
        return torch.sigmoid(self.mu * x)

    def forward(self, node_types, adj_mechanism=None, adj_het_structure=None):
        E = self.get_node_embeddings(node_types)
        Q_g, K_g, G_g = self.W_Q(E), self.W_K(E), self.W_G(E)


        scale = np.sqrt(self.node_embed_dim)
        a_G = self.modified_sigmoid((Q_g @ K_g.T) / scale)

        if adj_mechanism is not None:
            Gate = self.modified_sigmoid(G_g + self.beta * adj_mechanism)
        else:
            Gate = self.modified_sigmoid(G_g)

        raw_A = a_G * Gate

        if adj_het_structure is not None:
            A = adj_het_structure * raw_A.unsqueeze(0)
        else:
            A = raw_A.unsqueeze(0)

        return A
