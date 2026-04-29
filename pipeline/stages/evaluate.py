from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from pipeline.context import PipelineContext
from pipeline.models import ExecutionResult, QualityReport


def _to_array(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference - candidate) ** 2))
    if mse <= 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_gray = np.dot(reference[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
    candidate_gray = np.dot(candidate[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = gaussian_filter(reference_gray, sigma=1.5)
    mu_y = gaussian_filter(candidate_gray, sigma=1.5)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = gaussian_filter(reference_gray * reference_gray, sigma=1.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(candidate_gray * candidate_gray, sigma=1.5) - mu_y_sq
    sigma_xy = gaussian_filter(reference_gray * candidate_gray, sigma=1.5) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.clip(np.mean(numerator / (denominator + 1e-8)), -1.0, 1.0))


def _score_from_metrics(
    psnr_value: float,
    ssim_value: float,
    latency_seconds: float,
    timeout_seconds: int,
) -> float:
    timeout_seconds = max(1, int(timeout_seconds))
    latency_score = max(0.0, 1.0 - (latency_seconds / timeout_seconds))
    normalized_psnr = min(psnr_value / 40.0, 1.0)
    return float(np.mean([normalized_psnr, max(0.0, ssim_value), latency_score]))


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _extract_crop_ratio(scene_prompt: str) -> float:
    normalized = _normalize_text(scene_prompt)
    ratio = 0.5
    if any(keyword in normalized for keyword in ["30%", "0.3", "三成"]):
        ratio = 0.3
    elif any(keyword in normalized for keyword in ["40%", "0.4", "四成"]):
        ratio = 0.4
    elif any(keyword in normalized for keyword in ["50%", "0.5", "一半", "half"]):
        ratio = 0.5
    elif any(keyword in normalized for keyword in ["60%", "0.6", "六成"]):
        ratio = 0.6
    elif any(keyword in normalized for keyword in ["70%", "0.7", "七成"]):
        ratio = 0.7
    return max(0.05, min(ratio, 1.0))


def _build_target_image(reference_image: Image.Image, scene_prompt: str, output_size: tuple[int, int]) -> Image.Image:
    normalized = _normalize_text(scene_prompt)
    if any(keyword in normalized for keyword in ["crop", "center crop", "裁切", "裁剪", "居中", "中心", "提取"]):
        crop_ratio = _extract_crop_ratio(scene_prompt)
        crop_width = max(1, int(reference_image.width * crop_ratio))
        crop_height = max(1, int(reference_image.height * crop_ratio))
        left = max(0, (reference_image.width - crop_width) // 2)
        top = max(0, (reference_image.height - crop_height) // 2)
        target = reference_image.crop((left, top, left + crop_width, top + crop_height))
    elif any(keyword in normalized for keyword in ["super-resolution", "upscale", "放大", "超分"]):
        target = reference_image.resize(output_size, Image.Resampling.LANCZOS)
    else:
        target = reference_image

    if target.size != output_size:
        target = target.resize(output_size, Image.Resampling.LANCZOS)
    return target


def evaluate_stage(context: PipelineContext, execution_result: ExecutionResult) -> QualityReport:
    if not execution_result.output_image_path:
        raise FileNotFoundError("Execution did not produce an output image.")

    reference_image = Image.open(context.artifacts["input_image"])
    candidate_image = Image.open(execution_result.output_image_path)
    target_image = _build_target_image(
        reference_image, context.request.scene_prompt, candidate_image.size)
    reference_array = _to_array(target_image)
    candidate_array = _to_array(candidate_image, target_image.size)
    psnr_value = _psnr(reference_array, candidate_array)
    ssim_value = _ssim(reference_array, candidate_array)
    score = _score_from_metrics(
        psnr_value,
        ssim_value,
        execution_result.duration_seconds,
        context.config.executor_timeout_seconds,
    )

    report = QualityReport(
        psnr=psnr_value,
        ssim=ssim_value,
        latency_seconds=execution_result.duration_seconds,
        score=score,
        notes="Higher score indicates better alignment with the inferred task target and execution speed.",
    )
    context.write_json("quality.json", report.to_dict())
    return report
