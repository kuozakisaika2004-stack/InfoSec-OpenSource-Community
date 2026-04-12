from denoise_config import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

print("[INFO] 加载特征...")
X, y = [], []
for line in open(FEATURES_PATH):
    row = list(map(float, line.strip().split(",")))
    y.append(row[0])
    X.append(row[1:])

X = np.array(X)
y = np.array(y)

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.25, random_state=42)

print("[INFO] 训练随机森林...")
model = RandomForestRegressor(n_estimators=10, n_jobs=-1)
model.fit(trainX, trainY)

preds = model.predict(testX)
rmse = np.sqrt(mean_squared_error(testY, preds))
print(f"RMSE: {rmse:.4f}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("[INFO] 模型保存完成：denoiser.pickle")