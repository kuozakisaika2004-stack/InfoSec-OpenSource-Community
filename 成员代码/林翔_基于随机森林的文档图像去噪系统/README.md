# 基于随机森林的文档图像去噪系统
本项目实现基于随机森林回归算法的文档图像去噪功能。

## 项目功能
对含噪声的文档图像进行自动去噪处理，输出清晰干净的图像，可用于OCR前置增强。

## 运行环境
Python 3
OpenCV-Python
NumPy
scikit-learn

安装命令：
pip install opencv-python numpy scikit-learn

## 文件说明
denoise_config.py        项目路径与参数配置
helpers.py               图像预处理工具函数
build_features.py        构建5×5像素特征，生成features.csv
train_denoiser.py        训练随机森林去噪模型
denoise_document.py       对测试图像批量去噪并保存结果

## 使用步骤
1. 准备train、train_cleaned、test三个文件夹并放入对应图像
2. 运行 build_features.py 生成特征
3. 运行 train_denoiser.py 训练模型
4. 运行 denoise_document.py 批量去噪
5. 去噪结果保存在 test_cleaned 文件夹中

## 注意
所有图像需按文件名一一对应，保证带噪图像与干净图像文件名相同。