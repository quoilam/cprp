import os
from PIL import Image
import numpy as np

def center_crop(image, percentage):
    width, height = image.size
    crop_width = int(width * percentage / 100)
    crop_height = int(height * percentage / 100)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    return image.crop((left, top, right, bottom))

def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    with Image.open(image_path) as img:
        cropped_img = center_crop(img, 50)
        cropped_img.save(output_path)
    
    result = {
        'strategy_name': '图像中心裁剪',
        'output_path': output_path,
        'size': cropped_img.size
    }
    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract the central 50% of an image.")
    parser.add_argument('--image-path', type=str, help='Path to the input image.', required=True)
    parser.add_argument('--output-path', type=str, help='Path to save the output image.', required=True)
    parser.add_argument('--scene-prompt', type=str, help='Scene prompt for the operation.', required=True)

    args = parser.parse_args()

    result = run(args.image_path, args.output_path, args.scene_prompt)
    print(f"Image successfully processed. Output saved to {result['output_path']}.")
