from PIL import Image
import numpy as np
import cv2
from skimage import measure

def run(image_path: str, output_path: str, scene_prompt: str) -> None:
    # 读取图像
    image = Image.open(image_path)
    width, height = image.size
    center_percentage = 50
    
    # 计算裁剪区域
    new_width = int(width * center_percentage / 100)
    new_height = int(height * center_percentage / 100)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    
    # 保存裁剪后的图像
    cropped_image.save(output_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract the central 50% of the image and save it to a new file.")
    parser.add_argument("--image-path", type=str, help="Path to the input image", required=True)
    parser.add_argument("--output-path", type=str, help="Path to save the output image", required=True)
    parser.add_argument("--scene-prompt", type=str, help="Scene prompt for the operation", required=True)
    
    args = parser.parse_args()
    
    run(args.image_path, args.output_path, args.scene_prompt)
