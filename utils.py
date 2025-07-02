# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import cv2
from torchvision.transforms import functional as F

def load_image(path):
    """读取图像并转换为 PyTorch Tensor"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return F.to_tensor(img)

def extract_boolean_features(preds, vocab):
    """
    将 Faster R-CNN 输出 preds 转为布尔特征向量。
    preds: dict，包含 'labels'（Tensor[N]）
    vocab: list，所有可能的类别/谓词名称
    """
    features = np.zeros(len(vocab), dtype=np.uint8)
    labels = preds["labels"].cpu().numpy()
    for lab in labels:
        if 0 <= lab < len(vocab):
            features[lab] = 1
    return features

def save_numpy(arr, path):
    """保存 numpy 数组到 .npy 文件，自动创建目录"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def load_json(path):
    """加载并返回 JSON 对象"""
    with open(path, "r") as f:
        return json.load(f)
