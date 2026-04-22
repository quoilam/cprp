import cv2
import numpy as np
import math

# ---------------------------------------------------------------------------
# 评估函数 (Evaluation Metrics)
# ---------------------------------------------------------------------------

def calculate_mse(img1, img2):
    """
    均方误差 (Mean Squared Error, MSE)
    衡量预测值与真实值偏差的平方的期望值。
    **越小越好**
    """
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def calculate_psnr(img1, img2):
    """
    峰值信噪比 (Peak Signal-to-Noise Ratio, PSNR)
    衡量图像质量的客观标准，通常用于评估降噪或压缩算法。
    **越大越好** (单位: dB)
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

def calculate_mae(img1, img2):
    """
    平均绝对误差 (Mean Absolute Error, MAE)
    衡量预测值与真实值之间绝对误差的平均值。
    **越小越好**
    """
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

