# -*- coding: utf-8 -*-
import os
import torch

# 项目根目录自动定位
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# CLEVR 数据集目录（需包含 images/, questions/, scenes/）
DATA_DIR = os.path.join(BASE_DIR, "CLEVR_v1.0")
# 预处理输出目录
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
# 运行设备：GPU 或 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tsetlin Machine 超参数
TM_CONFIG = {
    "clauses": 1000,   # 子句数增至5000以覆盖高维特征空间
    "T": 50,           # 提高决策阈值
    "s": 1.5,          # 减小 s 值，降低对噪声的敏感度
    "epochs": 50       # 训练轮数
}