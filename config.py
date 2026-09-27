"""
Configuration for GAD-Net (Knowledge-Guided Adaptive Graph Network)
Steel Product Quality Prediction in CAPL Process
(连续退火产线钢铁产品质量预测)
"""
import os
import torch

# ======================== Path Configuration (路径配置) ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录作为项目根目录
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "CAPL.csv") # 主实验数据（GB18030 编码）
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")         # 实验结果（如CSV汇总表格）保存路径
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")         # 生成的图表保存路径
MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")     # 训练好的模型权重文件保存路径

# 自动检测并设置运行设备（如果有NVIDIA显卡且装了CUDA则用GPU，否则用CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 全局随机种子，确保每次运行的结果可复现
SEED = 42

# ======================== Figure Settings (图表生成设置) ========================
# 图片保存格式：同时输出PDF和SVG（方便放入论文或做矢量图修改）
FIG_FORMATS = ["pdf", "svg"]

# ======================== Column Name Mapping (数据列名中英映射) ============
# 将原始 Excel 数据里的中文列名映射为代码中统一使用的英文特征名
COLUMN_NAME_MAP = {
    "屈服强度":         "Yield_Strength",        # 预测目标
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
    "[C]":              "C_Content",             # 碳含量
    "[Mn]":             "Mn_Content",            # 锰含量
    "[S]":              "S_Content",             # 硫含量
    "[P]":              "P_Content",             # 磷含量
}
# 反向映射字典（备用）
COLUMN_NAME_MAP_INV = {v: k for k, v in COLUMN_NAME_MAP.items()}

# 绘图时使用的短特征名（为了防止图表坐标轴文字过长重叠）
SHORT_NAMES = {
    'Furnace_Speed': 'FS', 'JPF_Preheat_Temp': 'JPF-PT',
    'Heating_Furnace_Temp': 'HF-T', 'Soaking_Furnace_Temp': 'SF-T',
    'Slow_Cooling_Temp': 'SC-T', 'Fast_Cooling1_Temp': 'FC1-T',
    'Overaging_Furnace_Temp': 'OA-T', 'Fast_Cooling2_Temp': 'FC2-T',
    'Quenching_Temp': 'Q-T', 'Elongation': 'EL',
    'Rolling_Force': 'RF', 'Bending_Force': 'BF',
    'Heating_Temp': 'HT', 'Finish_Rolling_Temp': 'FRT',
    'Coiling_Temp': 'CT', 'Actual_Thickness': 'ATh',
    'Actual_Width': 'AWd', 'Cold_Rolling_Reduction': 'CRR',
    'C_Content': 'C', 'Mn_Content': 'Mn', 'S_Content': 'S', 'P_Content': 'P',
}

def short(name):
    """获取特征的缩写名，如果没有定义缩写则截取前12个字符。"""
    return SHORT_NAMES.get(name, name[:12])

# ======================== Data Configuration (数据集配置) ========================
TARGET_COL = "Yield_Strength"   # 模型的最终预测目标
TEST_RATIO = 0.15               # 测试集占比 15% (用于最终评估)
VAL_RATIO = 0.15                # 验证集占比 15% (用于训练过程中的早停和调参)
TRAIN_RATIO = 1 - TEST_RATIO - VAL_RATIO  # 剩下的 70% 作为训练集

# ======================== Graph Construction (图网络构建配置) ========================
# 给不同的工艺参数分配节点类型编号（0: 温度类, 1: 机械类, 2: 化学类, 3: 工艺尺寸类）
NODE_TYPE_MAP = {
    # 0: Operating, 1: Procedure, 2: Conditional, 3: Composition
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
NODE_TYPE_LABELS = {
    0: "Operating", 1: "Procedure", 2: "Conditional", 3: "Composition"
}
NUM_NODE_TYPES = 4              # 节点类型的总数
NUM_EDGE_TYPES = NUM_NODE_TYPES * NUM_NODE_TYPES  # 异构图中可能存在的边类型总数 (4x4=16种连接关系)

CORRELATION_THRESHOLD = 0.23    # 数据驱动建图时的皮尔逊相关性阈值（高于此值才认为两个特征相连）
MUTUAL_INFO_THRESHOLD = 0.10    # 互信息阈值（目前代码里作为图自环权重的下限保证）
GRAPH_SPARSITY_LAMBDA = 0.01    # 控制自适应学习图谱稀疏程度的正则化惩罚项
KNOWLEDGE_GATE_BETA = 0.9       # 知识门控权重（越大代表模型越“信任”人工整理的机理知识图谱）
GATE_SLOPE_MU = 1.8             # 门控函数的 Sigmoid 缩放斜率（控制阈值的平滑过渡程度）

# ======================== Model Hyperparameters (模型网络架构超参数) ========================
HIDDEN_DIM = 64                 # 神经网络隐藏层特征维度（控制模型容量）
NUM_GNN_LAYERS = 3              # 图神经网络（GNN）的层数/深度
NUM_ATTENTION_HEADS = 4         # 图注意力机制（GAT）的多头注意力头数
DROPOUT = 0.20                  # Dropout随机失活比例（防止过拟合）
NODE_EMBED_DIM = 24             # 特征节点嵌入向量的维度大小

# ======================== AdaBoost 超参数 (针对难样本/长尾样本) ========================
ADABOOST_NUM_ROUNDS = 5         # AdaBoost 的提升轮数
ADABOOST_LEARNING_RATE = 0.024   # AdaBoost 更新权重时的学习率
ADABOOST_TAIL_BOOST = 1.20      # 初始化时，直接给长尾（极端）样本放大的权重倍数
ADABOOST_TAIL_PERCENTILE = 10   # 尾部样本的定义：目标值排在最高 10% 或最低 10% 的样本

# 误差驱动尾部感知重加权：稳定化参数
ADABOOST_WARMUP_EPOCHS = 0     # 论文协议不设置预热期
ADABOOST_UPDATE_INTERVAL = 1   # 每个 epoch 更新一次尾部样本权重
ADABOOST_ERROR_EMA = 0.96       # 误差的指数移动平均系数
ADABOOST_MIN_TAIL_MULTIPLIER = 1.00 # 尾部样本权重的最低下限乘数
ADABOOST_MAX_TAIL_MULTIPLIER = 1.45  # 尾部权重上限
ADABOOST_ROBUST_QUANTILE = 80.0  # 0.80 quantile for robust scaling
ADABOOST_UPDATE_MOMENTUM = 0.90  # 尾部权重更新动量
ADABOOST_MEAN_ANCHOR = 0.04  # 尾部权重均值锚定系数

# ======================== Training (基础训练控制超参) ========================
BATCH_SIZE = 32                 # 批处理大小（每次喂入模型的样本数）
LEARNING_RATE = 1e-4            # 初始学习率 (0.0008)
LR_DECAY_STEP = 10              # 每隔 10 个 Epoch 衰减一次学习率
LR_DECAY_GAMMA = 0.8            # 每次衰减将学习率乘以 0.8
WEIGHT_DECAY = 1.5e-4           # L2 正则化权重衰减系数（防止过拟合）
NUM_EPOCHS = 300                # 最大训练轮数
EARLY_STOP_PATIENCE = 50         # 验证指标连续50轮不改善则停止
LOSS_MSE_WEIGHT = 1.0           # 预测主任务（均方误差）在总损失中的占比
LOSS_L1_WEIGHT = 0.001          # L1 稀疏正则化占比
LOSS_GRAPH_WEIGHT = 0.0008      # 辅助任务：图自适应学习结构的正则化损失占比

# ======================== Public experiment protocol ========================
HIT_RATE_THRESHOLD = 10
WEIGHT_VIS_INTERVAL = 100

# Basic physical admissibility bounds used before splitting.  Bounds are fixed
# by variable meaning and do not use sample quantiles, preventing split leakage.
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
