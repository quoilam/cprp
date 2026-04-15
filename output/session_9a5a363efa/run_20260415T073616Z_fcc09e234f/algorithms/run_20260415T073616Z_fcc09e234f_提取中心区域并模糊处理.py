import cv2
import numpy as np
from PIL import Image
import math

def extract_center_and_blur(image_path: str, output_path: str, center_percentage: int, blur_strength: int):
    # 读取图像
    image = Image.open(image_path)
    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    half_width = int(width * center_percentage / 100)
    half_height = int(height * center_percentage / 100)
    
    # 提取中心区域
    center_image = image.crop((center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height))
    
    # 转换为OpenCV格式
    center_cv_image = cv2.cvtColor(np.array(center_image), cv2.COLOR_RGB2BGR)
    
    # 模糊处理
    blurred_image = cv2.GaussianBlur(center_cv_image, (0, 0), sigmaX=blur_strength, sigmaY=blur_strength)
    
    # 旋转图像
    angle = 47
    rows, cols, _ = blurred_image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(blurred_image, M, (cols, rows))
    
    # 保存结果
    rotated_image_pil = Image.fromarray(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    rotated_image_pil.save(output_path)
    
    return {
        "strategy_name": "提取中心区域并模糊处理",
        "output_path": output_path,
        "size": rotated_image_pil.size
    }

def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    params = {
        "center_percentage": 50,
        "blur_strength": 80
    }
    result = extract_center_and_blur(image_path, output_path, **params)
    return result
