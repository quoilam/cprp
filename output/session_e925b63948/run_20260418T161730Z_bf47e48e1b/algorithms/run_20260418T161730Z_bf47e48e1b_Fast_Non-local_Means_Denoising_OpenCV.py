from typing import Dict
import cv2
import numpy as np
from pathlib import Path


def run(image_path: str, output_path: str, scene_prompt: str) -> Dict:
    """
    Apply Fast Non-local Means Denoising to the input image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str
        Path where the denoised image will be saved.
    scene_prompt : str
        Natural-language description of the task (kept for audit/traceability).

    Returns
    -------
    Dict
        {
            "strategy_name": str,
            "output_path": str,
            "size": (width, height),
            "scene_prompt": str
        }
    """
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at '{image_path}'")

    # Bilateral Filter (OpenCV) - edge-preserving denoising
    d = 11
    sigma_color = 100
    sigma_space = 25
    denoised = cv2.bilateralFilter(
        img,
        d=d,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write output
    written = cv2.imwrite(str(output_path), denoised)
    if not written:
        raise RuntimeError(f"Failed to write image to '{output_path}'")

    return {
        "strategy_name": "Bilateral Filter Denoising (OpenCV)",
        "output_path": str(output_path),
        "size": (int(img.shape[1]), int(img.shape[0])),
        "scene_prompt": scene_prompt
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Denoise an image using Fast Non-local Means Denoising.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    parser.add_argument("--output-path", required=True, help="Path where the denoised image will be saved.")
    parser.add_argument("--scene-prompt", required=True, help="Natural-language description of the task.")
    args = parser.parse_args()

    run(args.image_path, args.output_path, args.scene_prompt)


if __name__ == "__main__":
    main()
