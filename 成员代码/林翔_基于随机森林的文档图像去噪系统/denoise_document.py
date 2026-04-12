from denoise_config import *
from helpers import blur_and_threshold
import pickle
import cv2
import numpy as np
import os

# 自动创建输出文件夹
os.makedirs("test_cleaned", exist_ok=True)

print("[INFO] 加载模型...")
model = pickle.load(open(MODEL_PATH, "rb"))

testPath = "test"
for fname in sorted(os.listdir(testPath)):
    img = cv2.imread(os.path.join(testPath, fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orig = gray.copy()

    gray = cv2.copyMakeBorder(gray, 2,2,2,2, cv2.BORDER_REPLICATE)
    gray = blur_and_threshold(gray)

    feats = []
    h, w = gray.shape
    for y in range(h):
        for x in range(w):
            roi = gray[y:y+5, x:x+5]
            if roi.shape != (5,5):
                continue
            feats.append(roi.flatten())

    print(f"[INFO] 去噪中: {fname}")
    pixels = model.predict(feats)
    output = (pixels.reshape(orig.shape) * 255).astype("uint8")

    # 保存到 test_cleaned 文件夹，与test文件同名
    save_path = os.path.join("test_cleaned", fname)
    cv2.imwrite(save_path, output)

print("[INFO] 全部去噪完成，已保存到 test_cleaned 文件夹")