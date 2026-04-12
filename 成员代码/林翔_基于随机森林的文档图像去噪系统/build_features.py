from denoise_config import *
from helpers import blur_and_threshold
import os
import cv2
import random

# 按文件名排序，保证一一对应
train_files = sorted(os.listdir(TRAIN_PATH))
clean_files = sorted(os.listdir(CLEANED_PATH))

csv = open(FEATURES_PATH, "w")

print("[INFO] 构建特征中...")

for fname in train_files:
    trainPath = os.path.join(TRAIN_PATH, fname)
    cleanPath = os.path.join(CLEANED_PATH, fname)

    trainImage = cv2.imread(trainPath)
    cleanImage = cv2.imread(cleanPath)

    trainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2GRAY)
    cleanImage = cv2.cvtColor(cleanImage, cv2.COLOR_BGR2GRAY)

    # 扩充边界 2 像素
    trainImage = cv2.copyMakeBorder(trainImage, 2,2,2,2, cv2.BORDER_REPLICATE)
    cleanImage = cv2.copyMakeBorder(cleanImage, 2,2,2,2, cv2.BORDER_REPLICATE)

    trainImage = blur_and_threshold(trainImage)
    cleanImage = cleanImage.astype("float") / 255.0

    h, w = trainImage.shape
    for y in range(h):
        for x in range(w):
            roi = trainImage[y:y+5, x:x+5]
            if roi.shape != (5,5):
                continue

            feat = roi.flatten()
            target = cleanImage[y+2, x+2]

            if random.random() <= SAMPLE_PROB:
                line = [str(target)] + [str(f) for f in feat]
                csv.write(",".join(line) + "\n")

csv.close()
print("[INFO] 特征构建完成：features.csv")