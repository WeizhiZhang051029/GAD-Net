"""Configuration for the public GAD-Net experiment."""
import os
import torch


# Paths and runtime settings.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "CAPL.csv")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLUMN_NAME_MAP = {
    "屈服强度":         "Yield_Strength",
    "炉子速度":         "Furnace_Speed",
    "JPF（预热炉温度）": "JPF_Preheat_Temp",
    "加热炉温度":       "Heating_Furnace_Temp",
    "均热炉温度":       "Soaking_Furnace_Temp",
    "缓冷炉温度":       "Slow_Cooling_Temp",
    "快冷炉1温度":      "Fast_Cooling1_Temp",
    "过时效炉温度":     "Overaging_Furnace_Temp",
    "快冷炉2温度":      "Fast_Cooling2_Temp",
    "淬火炉温度":       "Quenching_Temp",
    "延伸率":           "Elongation",
    "轧制力":           "Rolling_Force",
    "弯辊力":           "Bending_Force",
    "加热温度":         "Heating_Temp",
    "终轧温度":         "Finish_Rolling_Temp",
    "卷取温度":         "Coiling_Temp",
    "实际厚度":         "Actual_Thickness",
    "实际宽度":         "Actual_Width",
    "冷轧压下率":       "Cold_Rolling_Reduction",
    "[C]":              "C_Content",
    "[Mn]":             "Mn_Content",
    "[S]":              "S_Content",
    "[P]":              "P_Content",
}

TARGET_COL = "Yield_Strength"


NODE_TYPE_MAP = {

    "Furnace_Speed": 0,
    "JPF_Preheat_Temp": 1, "Heating_Furnace_Temp": 1,
    "Soaking_Furnace_Temp": 1, "Slow_Cooling_Temp": 1,
    "Fast_Cooling1_Temp": 1, "Overaging_Furnace_Temp": 1,
    "Fast_Cooling2_Temp": 1, "Quenching_Temp": 1,
    "Heating_Temp": 1, "Finish_Rolling_Temp": 1,
    "Coiling_Temp": 1,
    "Elongation": 2, "Rolling_Force": 2, "Bending_Force": 2,
    "Actual_Thickness": 2, "Actual_Width": 2,
    "Cold_Rolling_Reduction": 2,
    "C_Content": 3, "Mn_Content": 3, "S_Content": 3, "P_Content": 3,
}
NUM_NODE_TYPES = 4
NUM_EDGE_TYPES = NUM_NODE_TYPES * NUM_NODE_TYPES

CORRELATION_THRESHOLD = 0.23
KNOWLEDGE_GATE_BETA = 0.9
GATE_SLOPE_MU = 1.8


HIDDEN_DIM = 64
NUM_GNN_LAYERS = 3
NUM_ATTENTION_HEADS = 4
DROPOUT = 0.20
NODE_EMBED_DIM = 24


ADABOOST_LEARNING_RATE = 0.024
ADABOOST_TAIL_BOOST = 1.20
ADABOOST_TAIL_PERCENTILE = 10


ADABOOST_ERROR_EMA = 0.96
ADABOOST_MIN_TAIL_MULTIPLIER = 1.00
ADABOOST_MAX_TAIL_MULTIPLIER = 1.45
ADABOOST_ROBUST_QUANTILE = 80.0
ADABOOST_UPDATE_MOMENTUM = 0.90
# Table 3 lists mean-anchor=0.04, but Algorithm 2 contains no anchoring step.
# Follow Algorithm 2 exactly: no mean anchoring is applied.


BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.8
WEIGHT_DECAY = 1.5e-4
NUM_EPOCHS = 300
EARLY_STOP_PATIENCE = 50
LOSS_MSE_WEIGHT = 1.0
LOSS_L1_WEIGHT = 0.001
LOSS_GRAPH_WEIGHT = 0.0008


HIT_RATE_THRESHOLD = 10


# Fixed domain bounds are applied before split generation.
PHYSICAL_BOUNDS = {
    "Yield_Strength": (0.0, None),
    "Furnace_Speed": (0.0, None),
    "JPF_Preheat_Temp": (0.0, None),
    "Heating_Furnace_Temp": (0.0, None),
    "Soaking_Furnace_Temp": (0.0, None),
    "Slow_Cooling_Temp": (0.0, None),
    "Fast_Cooling1_Temp": (0.0, None),
    "Overaging_Furnace_Temp": (0.0, None),
    "Fast_Cooling2_Temp": (0.0, None),
    "Quenching_Temp": (0.0, None),
    "Elongation": (0.0, None),
    "Rolling_Force": (0.0, None),
    "Heating_Temp": (0.0, None),
    "Finish_Rolling_Temp": (0.0, None),
    "Coiling_Temp": (0.0, None),
    "Actual_Thickness": (1e-9, None),
    "Actual_Width": (0.0, None),
    "Cold_Rolling_Reduction": (0.0, 1.0),
    "C_Content": (0.0, None), "Mn_Content": (0.0, None),
    "S_Content": (0.0, None), "P_Content": (0.0, None),
}
