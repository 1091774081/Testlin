# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from config import DATA_DIR, OUTPUT_DIR
from utils import save_numpy, load_json

# 定义特征 vocab：shape, color, size, material, spatial and vertical relations, count features
shapes = ['cube', 'sphere', 'cylinder']
colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'yellow']
sizes = ['small', 'large']
materials = ['rubber', 'metal']
relations = ['left_of', 'right_of', 'front_of', 'behind', 'above', 'below']
# 计数特征：针对 shape-color-size 组合，是否>=2
count_predicates = [f"{s}_{c}_{z}_count>=2" for s in shapes for c in colors for z in sizes]
# 总 vocab
vocab = shapes + colors + sizes + materials + relations + count_predicates

# 特征提取函数
def extract_scene_features(objs):
    """根据 scene 对象列表提取布尔特征向量"""
    features = np.zeros(len(vocab), dtype=np.uint8)
    # 单体属性
    for obj in objs:
        s, c, z, m = obj.get('shape'), obj.get('color'), obj.get('size'), obj.get('material')
        if s in shapes:
            features[shapes.index(s)] = 1
        if c in colors:
            features[len(shapes) + colors.index(c)] = 1
        if z in sizes:
            features[len(shapes) + len(colors) + sizes.index(z)] = 1
        if m in materials:
            features[len(shapes) + len(colors) + len(sizes) + materials.index(m)] = 1
    # 计数特征
    counts = defaultdict(int)
    for obj in objs:
        key = (obj.get('shape'), obj.get('color'), obj.get('size'))
        counts[key] += 1
    for idx, pred in enumerate(count_predicates, start=len(shapes)+len(colors)+len(sizes)+len(materials)+len(relations)):
        s, c, z, _ = pred.split('_')
        if counts.get((s, c, z), 0) >= 2:
            features[idx] = 1
    # 空间和垂直关系
    for i, obj1 in enumerate(objs):
        for obj2 in objs[i+1:]:
            x1, y1, z1 = obj1['3d_coords']
            x2, y2, z2 = obj2['3d_coords']
            base = len(shapes)+len(colors)+len(sizes)+len(materials)
            # left/right
            if x1 < x2:
                features[base + relations.index('left_of')] = 1
            elif x1 > x2:
                features[base + relations.index('right_of')] = 1
            # front/behind (z)
            if z1 < z2:
                features[base + relations.index('behind')] = 1
            elif z1 > z2:
                features[base + relations.index('front_of')] = 1
            # above/below (y)
            if y1 > y2:
                features[base + relations.index('above')] = 1
            elif y1 < y2:
                features[base + relations.index('below')] = 1
    return features

# 处理函数，只处理 train 和 val

def process_split(split, max_samples=None):
    """处理 train/val 的特征与标签并保存"""
    qfile = os.path.join(DATA_DIR, 'questions', f'CLEVR_{split}_questions.json')
    questions = load_json(qfile)['questions']
    scene_file = os.path.join(DATA_DIR, 'scenes', f'CLEVR_{split}_scenes.json')
    scenes = load_json(scene_file)['scenes']
    scene_map = {sce['image_filename']: sce['objects'] for sce in scenes}

    if max_samples and len(questions) > max_samples:
        questions = questions[:max_samples]
    N = len(questions)

    X = np.zeros((N, len(vocab)), dtype=np.uint8)
    Y = np.zeros(N, dtype=np.int64)

    for i, q in enumerate(tqdm(questions, desc=split)):
        objs = scene_map.get(q['image_filename'], [])
        X[i] = extract_scene_features(objs)
        ans = q.get('answer', '')
        Y[i] = int(ans) if ans.isdigit() and 0 <= int(ans) < 10 else -1

    idx = np.where(Y >= 0)[0]
    X = X[idx]
    Y = Y[idx]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_numpy(X, os.path.join(OUTPUT_DIR, f'X_{split}.npy'))
    save_numpy(Y, os.path.join(OUTPUT_DIR, f'Y_{split}.npy'))

# 主入口
if __name__ == '__main__':
    max_samples = 20000
    for split in ['train', 'val']:
        print(f"Processing {split}, limit to {max_samples} samples")
        process_split(split, max_samples)