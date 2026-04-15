import os
from PIL import Image
import numpy as np
import argparse

def center_crop(image, crop_size):
    width, height = image.size
    crop_width = int(width * float(crop_size.replace('%', '')) / 100)
    crop_height = int(height * float(crop_size.replace('%', '')) / 100)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = (width + crop_width) // 2
    bottom = (height + crop_height) // 2
    return image.crop((left, top, right, bottom))

def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    image = Image.open(image_path)
    crop_size = '50%'
    cropped_image = center_crop(image, crop_size)
    cropped_image.save(output_path)
    result = {
        "strategy_name": "CenterCrop",
        "output_path": output_path,
        "size": cropped_image.size
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the central 50% of an image.")
    parser.add_argument('--image-path', type=str, help='Path to the input image')
    parser.add_argument('--output-path', type=str, help='Path to save the output image')
    parser.add_argument('--scene-prompt', type=str, help='Scene prompt for the operation')
    
    args = parser.parse_args()
    
    if not all([args.image_path, args.output_path, args.scene_prompt]):
        parser.print_help()
        exit(1)
    
    result = run(args.image_path, args.output_path, args.scene_prompt)
    print(f"Image saved to {result['output_path']} with size {result['size']}")
