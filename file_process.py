import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# ---------------- 配置 ---------------- #
input_root = "D:/face_expression_self/facial_expression_dataset/facial_expression_dataset/test"
output_root = "./fer2013_augmented_test/train"
face_size = (48, 48)
num_augments_per_image = 3  # 每张图像增强次数

# ---------------- 增强函数 ---------------- #
def augment_image(img):
    """
    对图像进行多种增强（随机组合）
    """
    h, w = img.shape

    # 随机水平翻转
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # 随机旋转
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 随机缩放
    scale = random.uniform(0.9, 1.1)
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    center = resized.shape[0] // 2
    if resized.shape[0] >= h:
        img = resized[center - h // 2:center + h // 2, center - w // 2:center + w // 2]
    else:
        pad = ((h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2)
        img = cv2.copyMakeBorder(resized, pad[0], pad[0], pad[1], pad[1], cv2.BORDER_REFLECT)

    # 随机亮度变化
    delta = random.randint(-30, 30)
    img = np.clip(img + delta, 0, 255).astype(np.uint8)

    return img

# ---------------- 主处理流程 ---------------- #
for cls in sorted(os.listdir(input_root)):
    in_dir = os.path.join(input_root, cls)
    out_dir = os.path.join(output_root, cls)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(in_dir):
        continue

    for fname in tqdm(os.listdir(in_dir), desc=cls):
        fpath = os.path.join(in_dir, fname)
        img = cv2.imread(fpath)

        if img is None:
            print("坏图像：", fpath)
            continue

        # --- 灰度化 + 均衡化 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, face_size, interpolation=cv2.INTER_CUBIC)

        # 保存原始处理图像
        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base_name}.png"), gray)

        # --- 图像增强 ---
        for i in range(num_augments_per_image):
            aug_img = augment_image(gray.copy())
            aug_name = f"{base_name}_aug{i}.png"
            cv2.imwrite(os.path.join(out_dir, aug_name), aug_img)

print("✅ 所有图像处理与增强完成！")


