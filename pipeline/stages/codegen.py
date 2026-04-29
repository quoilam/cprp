from __future__ import annotations

import json
from hashlib import sha256

from prompts import CODEGEN_PROMPT_TEMPLATE, PREPARE_CODEGEN_PROMPT_TEMPLATE
from pipeline.context import PipelineContext
from pipeline.models import CandidateMethod, GeneratedAlgorithmArtifact, ResearchResult

from .common import (
    call_openrouter_code,
    safe_filename_fragment,
    validate_allowed_imports,
    validate_prepare_signature,
    validate_python_source,
    validate_run_signature,
    verify_generated_algorithm_contract,
)
from .research import format_web_context


def _normalize_strategy_key(strategy: str, description: str = "", scene_prompt: str = "") -> str:
    text = " ".join([strategy, description, scene_prompt]).lower()
    if any(keyword in text for keyword in ["crop", "裁切", "裁剪", "居中", "中心", "提取"]):
        return "center_crop"
    if any(keyword in text for keyword in ["histogram", "equalization", "直方图", "均衡"]):
        return "histogram_equalization"
    if any(keyword in text for keyword in ["laplacian", "拉普拉斯", "锐化"]):
        return "laplacian_sharpen"
    if any(keyword in text for keyword in ["fusion", "融合", "contrast", "对比度", "增强", "enhance", "清晰"]):
        return "fusion_contrast_enhancement"
    if any(keyword in text for keyword in ["denoise", "noise", "去噪", "降噪"]):
        return "denoise_conservative"
    if any(keyword in text for keyword in ["deblur", "blur", "去模糊"]):
        return "deblur_sharpen"
    if any(keyword in text for keyword in ["super-resolution", "upscale", "放大", "超分"]):
        return "upscale_refine"
    return "enhance_contrast"


def _find_chosen_candidate(research_result: ResearchResult) -> CandidateMethod | None:
    chosen = research_result.chosen_strategy.strip().lower()
    for candidate in research_result.candidates:
        if candidate.name.strip().lower() == chosen:
            return candidate
    if research_result.candidates:
        chosen_key = _normalize_strategy_key(
            research_result.chosen_strategy, scene_prompt=research_result.scene_prompt)
        for candidate in research_result.candidates:
            candidate_key = _normalize_strategy_key(
                candidate.name, candidate.description, research_result.scene_prompt)
            if candidate_key == chosen_key:
                return candidate
        return research_result.candidates[0]
    return None


def _generate_algorithm_source_with_strategy(strategy: str, strategy_key: str, parameters: dict) -> str:
    serialized_parameters = json.dumps(
        parameters, ensure_ascii=False, indent=4)
    return f"""from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


STRATEGY_NAME = {strategy!r}
STRATEGY_KEY = {strategy_key!r}
STRATEGY_PARAMETERS = {serialized_parameters}


def _parameter_float(name: str, default: float) -> float:
    try:
        return float(STRATEGY_PARAMETERS.get(name, default))
    except (TypeError, ValueError):
        return default


def _natural_enhance(image: Image.Image, contrast: float = 1.12, sharpness: float = 1.08) -> Image.Image:
    processed = ImageOps.autocontrast(image, cutoff=1)
    processed = ImageEnhance.Contrast(processed).enhance(contrast)
    processed = ImageEnhance.Sharpness(processed).enhance(sharpness)
    return processed


def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    source_path = Path(image_path)
    destination_path = Path(output_path)
    image = Image.open(source_path).convert(\"RGB\")

    if STRATEGY_KEY == \"denoise_conservative\":
        processed = image.filter(ImageFilter.MedianFilter(size=int(STRATEGY_PARAMETERS.get(\"filter_size\", 3))))
        processed = _natural_enhance(processed, contrast=_parameter_float(\"contrast\", 1.05), sharpness=_parameter_float(\"sharpness\", 1.03))
    elif STRATEGY_KEY == \"center_crop\":
        crop_ratio = float(STRATEGY_PARAMETERS.get(\"crop_ratio\", 0.5))
        crop_ratio = max(0.05, min(crop_ratio, 1.0))
        crop_width = max(1, int(image.width * crop_ratio))
        crop_height = max(1, int(image.height * crop_ratio))
        left = max(0, (image.width - crop_width) // 2)
        top = max(0, (image.height - crop_height) // 2)
        processed = image.crop((left, top, left + crop_width, top + crop_height))
    elif STRATEGY_KEY in (\"deblur_sharpen\", \"laplacian_sharpen\"):
        processed = image.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
        processed = _natural_enhance(processed, contrast=_parameter_float(\"contrast\", 1.06), sharpness=_parameter_float(\"sharpness\", 1.14))
    elif STRATEGY_KEY == \"upscale_refine\":
        scale = int(STRATEGY_PARAMETERS.get(\"scale\", 2))
        processed = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
        processed = _natural_enhance(processed, contrast=_parameter_float(\"contrast\", 1.04), sharpness=_parameter_float(\"sharpness\", 1.08))
    elif STRATEGY_KEY == \"histogram_equalization\":
        ycbcr = image.convert(\"YCbCr\")
        y, cb, cr = ycbcr.split()
        y = ImageOps.equalize(y)
        processed = Image.merge(\"YCbCr\", (y, cb, cr)).convert(\"RGB\")
        processed = ImageEnhance.Sharpness(processed).enhance(_parameter_float(\"sharpness\", 1.06))
    elif STRATEGY_KEY in (\"enhance_contrast\", \"fusion_contrast_enhancement\"):
        processed = _natural_enhance(
            image,
            contrast=_parameter_float(\"contrast\", 1.14),
            sharpness=_parameter_float(\"sharpness\", 1.10),
        )
    else:
        processed = _natural_enhance(image)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(destination_path)
    return {{
        \"strategy_name\": STRATEGY_NAME,
        \"strategy_key\": STRATEGY_KEY,
        \"scene_prompt\": scene_prompt,
        \"output_path\": str(destination_path),
        \"size\": [processed.width, processed.height],
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--image-path\", required=True)
    parser.add_argument(\"--output-path\", required=True)
    parser.add_argument(\"--scene-prompt\", required=True)
    args = parser.parse_args()
    payload = run(args.image_path, args.output_path, args.scene_prompt)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == \"__main__\":
    main()
"""


def _generate_algorithm_source(research_result: ResearchResult) -> str:
    candidate = _find_chosen_candidate(research_result)
    strategy = research_result.chosen_strategy or (candidate.name if candidate else "enhance_contrast")
    description = candidate.description if candidate else research_result.summary
    parameters = candidate.parameters if candidate else {}
    strategy_key = _normalize_strategy_key(
        strategy, description, research_result.scene_prompt)
    return _generate_algorithm_source_with_strategy(strategy, strategy_key, parameters)


def _generate_prepare_source() -> str:
    return """from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image


def evaluate(input_image_path: str, output_image_path: str) -> Dict[str, float]:
    src = np.asarray(Image.open(input_image_path).convert(\"RGB\"), dtype=np.float32)
    dst = np.asarray(Image.open(output_image_path).convert(\"RGB\"), dtype=np.float32)
    if src.shape != dst.shape:
        dst = np.asarray(Image.open(output_image_path).convert(\"RGB\").resize((src.shape[1], src.shape[0])), dtype=np.float32)

    mse = float(np.mean((src - dst) ** 2))
    psnr = 99.0 if mse <= 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return {\"psnr\": psnr}
"""


def codegen_stage(context: PipelineContext, research_result: ResearchResult) -> GeneratedAlgorithmArtifact:
    algo_dir = context.paths.generated_algorithms_dir
    algo_dir.mkdir(parents=True, exist_ok=True)
    strategy_fragment = safe_filename_fragment(research_result.chosen_strategy)
    algorithm_path = algo_dir / f"{context.run_id}_{strategy_fragment}.py"
    prepare_path = algo_dir / \
        f"{context.run_id}_{strategy_fragment}.prepare.py"

    research_summary = json.dumps(
        research_result.to_dict(), ensure_ascii=False, indent=2)
    web_clues_path = context.paths.run_dir / "research_web_clues.json"
    web_clues_payload = {}
    if web_clues_path.exists():
        web_clues_payload = json.loads(
            web_clues_path.read_text(encoding="utf-8"))
    web_context = format_web_context(web_clues_payload.get("clues") or [])

    prompt = CODEGEN_PROMPT_TEMPLATE.format(
        research_summary=research_summary, web_context=web_context)
    source = ""
    last_error = ""
    if context.config.bypass_autoresearch:
        context.log_event(
            "codegen",
            "bypass_autoresearch",
            {"strategy": research_result.chosen_strategy, "reason": "Using deterministic local generator."},
        )
        source = _generate_algorithm_source(research_result)
        validate_python_source(source)
        validate_run_signature(source)
        validate_allowed_imports(source)
        algorithm_path.write_text(source, encoding="utf-8")
        ok, message = verify_generated_algorithm_contract(
            context=context,
            algorithm_path=algorithm_path,
            image_path=context.artifacts["input_image"],
            scene_prompt=context.request.scene_prompt,
            verify_dir=context.paths.artifacts_dir / "codegen_verify",
        )
        if not ok:
            raise RuntimeError(
                f"Deterministic local algorithm failed contract verification: {message}"
            )
    else:
        for attempt in range(3):
            context.log_event("codegen", "attempt", {"attempt": attempt + 1})
            try:
                source = call_openrouter_code(context, prompt)
                validate_python_source(source)
                validate_run_signature(source)
                validate_allowed_imports(source)
                algorithm_path.write_text(source, encoding="utf-8")
                ok, message = verify_generated_algorithm_contract(
                    context=context,
                    algorithm_path=algorithm_path,
                    image_path=context.artifacts["input_image"],
                    scene_prompt=context.request.scene_prompt,
                    verify_dir=context.paths.artifacts_dir / "codegen_verify",
                )
                if ok:
                    break
                last_error = message
            except Exception as exc:
                last_error = str(exc)
                context.log_event("codegen", "attempt_error", {
                                  "attempt": attempt + 1, "error": last_error[:200]})

            prompt = (
                "Rewrite the COMPLETE Python file to satisfy contract.\n"
                f"Previous error: {last_error}\n"
                "STRICT constraints:\n"
                "- Allowed libraries: Python stdlib, Pillow(PIL), numpy, scipy, cv2, skimage, imageio.\n"
                "- Must define run(image_path, output_path, scene_prompt) and save image to output_path.\n"
                "- Keep a runnable CLI main with --image-path --output-path --scene-prompt.\n"
                "- Output Python code only.\n"
                "Research summary:\n"
                f"{research_summary}\n"
                "Web clues:\n"
                f"{web_context}\n"
                "Previous code:\n"
                f"```python\n{source}\n```"
            )
        else:
            context.log_event(
                "codegen",
                "fallback_local_generator",
                {"reason": last_error[:200],
                    "strategy": research_result.chosen_strategy},
            )
            source = _generate_algorithm_source(research_result)
            validate_python_source(source)
            validate_run_signature(source)
            validate_allowed_imports(source)
            algorithm_path.write_text(source, encoding="utf-8")
            ok, message = verify_generated_algorithm_contract(
                context=context,
                algorithm_path=algorithm_path,
                image_path=context.artifacts["input_image"],
                scene_prompt=context.request.scene_prompt,
                verify_dir=context.paths.artifacts_dir / "codegen_verify",
            )
            if not ok:
                raise RuntimeError(
                    f"Generated algorithm failed contract verification: {last_error}; local fallback failed: {message}"
                )

    prepare_prompt = PREPARE_CODEGEN_PROMPT_TEMPLATE.format(
        research_summary=research_summary, web_context=web_context)
    prepare_source = ""
    prepare_last_error = ""
    if context.config.bypass_autoresearch:
        prepare_source = _generate_prepare_source()
        validate_python_source(prepare_source)
        validate_prepare_signature(prepare_source)
        validate_allowed_imports(prepare_source)
        prepare_path.write_text(prepare_source, encoding="utf-8")
    else:
        for attempt in range(3):
            context.log_event("codegen", "prepare_attempt",
                              {"attempt": attempt + 1})
            try:
                prepare_source = call_openrouter_code(context, prepare_prompt)
                validate_python_source(prepare_source)
                validate_prepare_signature(prepare_source)
                validate_allowed_imports(prepare_source)
                prepare_path.write_text(prepare_source, encoding="utf-8")
                break
            except Exception as exc:
                prepare_last_error = str(exc)
                prepare_prompt = (
                    "Rewrite the COMPLETE prepare.py file to satisfy contract.\n"
                    f"Previous error: {prepare_last_error}\n"
                    "STRICT constraints:\n"
                    "- Allowed libraries: Python stdlib, Pillow(PIL), numpy, scipy, cv2, skimage, imageio.\n"
                    "- Must define evaluate(input_image_path, output_image_path) returning dict with metric fields.\n"
                    "- Output Python code only.\n"
                    "Research summary:\n"
                    f"{research_summary}\n"
                    "Web clues:\n"
                    f"{web_context}\n"
                    "Previous code:\n"
                    f"```python\n{prepare_source}\n```"
                )
        else:
            context.log_event(
                "codegen",
                "prepare_fallback_local_generator",
                {"reason": prepare_last_error[:200]},
            )
            prepare_source = _generate_prepare_source()
            validate_python_source(prepare_source)
            validate_prepare_signature(prepare_source)
            validate_allowed_imports(prepare_source)
            prepare_path.write_text(prepare_source, encoding="utf-8")

    artifact = GeneratedAlgorithmArtifact(
        path=algorithm_path,
        prepare_path=prepare_path,
        source_hash=sha256(source.encode("utf-8")).hexdigest(),
        strategy_name=research_result.chosen_strategy,
        syntax_validated=True,
    )
    context.artifacts["algorithm"] = algorithm_path
    context.artifacts["prepare"] = prepare_path
    context.write_json("codegen.json", artifact.to_dict())
    return artifact
