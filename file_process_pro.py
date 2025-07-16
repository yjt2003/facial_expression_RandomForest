import os
from pathlib import Path
import cv2
from cv2 import dnn_superres
from tqdm.auto import tqdm

# ---------- 参数 ----------
INPUT_DIR  = Path("D:\\face_expression_self\\fer2013_augmented_train\\train")       # 已有的 48×48 图像根目录
OUTPUT_DIR = Path("fer2013_sr_192x192_train")   # 超分后输出目录
MODEL_PATH = Path("D:\\face_expression_self\\ESPCN_x4.pb")        # 超分模型
SCALE      = 4                          # 放大倍数 (2× -> 96×96)
IMG_SUFFIX = (".png", ".jpg", ".jpeg")

# ---------- 初始化超分模型 ----------
sr = dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(MODEL_PATH))
sr.setModel("espcn", SCALE)

# ---------- 遍历并处理 ----------
img_paths = list(INPUT_DIR.rglob("*"))
img_paths = [p for p in img_paths if p.suffix.lower() in IMG_SUFFIX]

pbar = tqdm(img_paths, desc="Super‑Resolution")
for src_path in pbar:
    # 读灰度图
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ 跳过无法读取的文件: {src_path}")
        continue

    # 超分
    up_img = sr.upsample(img)

    # 生成目标路径（保持子目录结构）
    rel_path = src_path.relative_to(INPUT_DIR)
    dst_path = OUTPUT_DIR / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 (仍保持灰度单通道)
    cv2.imwrite(str(dst_path), up_img)

print(f"✅ 全部完成，输出目录: {OUTPUT_DIR}")
