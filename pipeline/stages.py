from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import ast
import importlib.util
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import gaussian_filter
from openai import OpenAI

from prompts import (
    CODEGEN_PROMPT_TEMPLATE,
    PIPELINE_SYSTEM_PROMPT,
    PREPARE_CODEGEN_PROMPT_TEMPLATE,
    RESEARCH_PROMPT_TEMPLATE,
)

from .context import PipelineContext
from .models import (
    CandidateMethod,
    DEFAULT_HTTP_MAX_RETRIES,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_MEMORY_MB,
    DEFAULT_OPTIMIZER_FUNCTION,
    DEFAULT_OPTIMIZER_MODULE,
    DEFAULT_RETRY_INITIAL_DELAY,
    DEFAULT_RETRY_JITTER,
    DEFAULT_RETRY_MAX_DELAY,
    DEFAULT_TIMEOUT_SECONDS,
    ExecutionResult,
    GeneratedAlgorithmArtifact,
    QualityReport,
    ResearchResult,
)
from .resilience import (
    BusinessLogicError,
    ModelResponseParseError,
    RetryableStatusCodeError,
    build_retry_policy,
    build_retry_decorator,
    extract_status_code,
)


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _choose_strategy(scene_prompt: str) -> tuple[str, dict[str, Any], str]:
    normalized = _normalize_text(scene_prompt)
    crop_intent = any(keyword in normalized for keyword in [
                      "crop", "center crop", "裁切", "裁剪", "居中", "中心", "提取"])
    if crop_intent:
        crop_ratio = 0.5
        if any(keyword in normalized for keyword in ["30%", "0.3", "三成"]):
            crop_ratio = 0.3
        elif any(keyword in normalized for keyword in ["40%", "0.4", "四成"]):
            crop_ratio = 0.4
        elif any(keyword in normalized for keyword in ["50%", "0.5", "一半", "半", "half"]):
            crop_ratio = 0.5
        elif any(keyword in normalized for keyword in ["60%", "0.6", "六成"]):
            crop_ratio = 0.6
        elif any(keyword in normalized for keyword in ["70%", "0.7", "七成"]):
            crop_ratio = 0.7
        return "center_crop", {"crop_ratio": crop_ratio}, "Prompt suggests center crop extraction"
    if any(keyword in normalized for keyword in ["denoise", "noise", "去噪"]):
        return "denoise_conservative", {"filter_size": 3, "contrast": 1.03}, "Prompt suggests noise reduction"
    if any(keyword in normalized for keyword in ["deblur", "blur", "去模糊"]):
        return "deblur_sharpen", {"sharpness": 1.35, "detail": 1.08}, "Prompt suggests blur correction"
    if any(keyword in normalized for keyword in ["super-resolution", "upscale", "放大", "超分"]):
        return "upscale_refine", {"scale": 2, "sharpness": 1.1}, "Prompt suggests upscaling"
    if any(keyword in normalized for keyword in ["enhance", "contrast", "clear", "增强", "清晰"]):
        return "enhance_contrast", {"contrast": 1.15, "sharpness": 1.08}, "Prompt suggests global enhancement"
    return "balanced_enhancement", {"contrast": 1.08, "sharpness": 1.05}, "Default balanced image enhancement"


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in model response.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object response.")
    return payload


def _looks_like_truncated_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith(("...", "```", ",", ":", "\\")):
        return True
    if stripped.count("{") > stripped.count("}"):
        return True
    if stripped.count("[") > stripped.count("]"):
        return True
    if stripped.startswith("{") and not stripped.endswith("}"):
        return True
    return False


def _extract_python_code(text: str) -> str:
    text = text.strip()
    code_block = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if code_block:
        return code_block.group(1).strip() + "\n"
    return text + ("\n" if not text.endswith("\n") else "")


def _safe_filename_fragment(value: str) -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_",
                     value, flags=re.UNICODE).strip("_")
    return cleaned or "generated_algorithm"


def _create_openrouter_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for research/codegen.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is required for research/codegen.")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _call_openrouter_json(context: PipelineContext, prompt: str) -> dict[str, Any]:
    start_time = time.time()
    policy = build_retry_policy(
        max_retries=DEFAULT_LLM_MAX_RETRIES,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY,
        jitter=DEFAULT_RETRY_JITTER,
    )
    llm_timeout = float(DEFAULT_LLM_TIMEOUT_SECONDS)
    context.log_event("tool_call", "openrouter_json_start",
                      {
                          "prompt_preview": prompt[:240],
                          "timeout_seconds": llm_timeout,
                          "max_retries": policy.max_retries,
                          "max_attempts": policy.max_attempts,
                      })
    client, model = _create_openrouter_client()

    @build_retry_decorator(
        context=context,
        stage="research",
        operation="openrouter_json",
        policy=policy,
    )
    def _call_once() -> tuple[str, dict[str, Any]]:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=llm_timeout,
        )
        content = completion.choices[0].message.content or ""
        try:
            return content, _extract_json_object(content)
        except Exception as exc:
            is_truncated = _looks_like_truncated_json(content)
            message = (
                "OpenRouter JSON parse failed: response appears truncated."
                if is_truncated
                else "OpenRouter JSON parse failed: invalid JSON payload."
            )
            raise ModelResponseParseError(
                message,
                retryable=is_truncated,
            ) from exc

    content, payload = _call_once()
    duration = time.time() - start_time
    context.log_event("tool_call", "openrouter_json_finish", {
                      "model": model, "duration_seconds": duration, "response_preview": content[:240]})
    return payload


def _call_openrouter_code(context: PipelineContext, prompt: str) -> str:
    start_time = time.time()
    context.log_event("tool_call", "openrouter_code_start",
                      {"prompt_preview": prompt[:240]})
    client, model = _create_openrouter_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    duration = time.time() - start_time
    context.log_event("tool_call", "openrouter_code_finish", {
                      "model": model, "duration_seconds": duration, "response_preview": content[:240]})
    return _extract_python_code(content)


def _search_web_clues(context: PipelineContext, scene_prompt: str) -> list[dict[str, str]]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        context.log_event("tool_call", "tavily_skip", {
                          "reason": "TAVILY_API_KEY missing"})
        return []
    query_plan = [
        (
            "algorithm",
            (
                "image processing implementation methods for scene: "
                f"{scene_prompt}. focus on practical algorithm steps and parameter choices"
            ),
        ),
        (
            "evaluation",
            (
                "image processing quality evaluation metrics for scene: "
                f"{scene_prompt}. focus on objective metrics and scoring protocol"
            ),
        ),
    ]
    start_time = time.time()
    policy = build_retry_policy(
        max_retries=DEFAULT_HTTP_MAX_RETRIES,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY,
        jitter=DEFAULT_RETRY_JITTER,
    )
    http_timeout = float(DEFAULT_HTTP_TIMEOUT_SECONDS)
    context.log_event("tool_call", "tavily_start", {
                      "query": f"image processing algorithm for: {scene_prompt}",
                      "timeout_seconds": http_timeout,
                      "max_retries": policy.max_retries,
                      "max_attempts": policy.max_attempts,
                      })

    @build_retry_decorator(
        context=context,
        stage="research",
        operation="tavily_search",
        policy=policy,
    )
    def _request_once() -> list[dict[str, str]]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": f"image processing algorithm for: {scene_prompt}",
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_images": False,
            },
            timeout=http_timeout,
        )
        status_code = int(response.status_code)

        if 400 <= status_code < 500 and status_code != 429:
            context.log_event("tool_call", "tavily_fast_fail", {
                              "status_code": status_code, "degraded": True})
            return []

        if status_code == 429 or status_code >= 500:
            raise RetryableStatusCodeError(
                status_code,
                f"Tavily retryable status: {status_code}",
            )

        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        clues: list[dict[str, str]] = []
        for item in results[:5]:
            clues.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "content": str(item.get("content") or "")[:500],
                }
            )
        return clues

    try:
        clues = _request_once()
        context.log_event("tool_call", "tavily_finish", {
                          "duration_seconds": time.time() - start_time, "result_count": len(clues)})
        return clues
    except Exception as exc:
        status_code = extract_status_code(exc)
        context.log_event("tool_call", "tavily_error", {
                          "duration_seconds": time.time() - start_time,
                          "error": str(exc),
                          "error_type": type(exc).__name__,
                          "status_code": status_code,
                          "degraded": True,
                          })
        return []


def _format_web_context(clues: list[dict[str, str]]) -> str:
    if not clues:
        return "(no external web clues available)"
    lines: list[str] = []
    for idx, clue in enumerate(clues, 1):
        topic = str(clue.get("topic") or "general").strip()
        query = str(clue.get("query") or "").strip()
        lines.append(f"Topic: {topic}")
        if query:
            lines.append(f"Query: {query}")
        lines.append(f"[{idx}] {clue.get('title', '').strip()}")
        lines.append(f"URL: {clue.get('url', '').strip()}")
        lines.append(f"Summary: {clue.get('content', '').strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def _validate_run_signature(source: str) -> None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names[:3] == ["image_path", "output_path", "scene_prompt"]:
                return
            raise ValueError(
                "run() must have arguments (image_path, output_path, scene_prompt).")
    raise ValueError(
        "Generated algorithm is missing run(image_path, output_path, scene_prompt).")


def _validate_allowed_imports(source: str) -> None:
    allowed_third_party = {"numpy", "PIL",
                           "scipy", "cv2", "skimage", "imageio"}
    stdlib_roots = set(getattr(sys, "stdlib_module_names", set()))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in stdlib_roots:
                    continue
                if root not in allowed_third_party:
                    raise ValueError(
                        f"Disallowed third-party import: {alias.name}")
                if importlib.util.find_spec(root) is None:
                    raise ValueError(
                        f"Unavailable import in environment: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in stdlib_roots:
                continue
            if root not in allowed_third_party:
                raise ValueError(
                    f"Disallowed third-party import: {node.module}")
            if importlib.util.find_spec(root) is None:
                raise ValueError(
                    f"Unavailable import in environment: {node.module}")


def _verify_generated_algorithm_contract(
    context: PipelineContext,
    algorithm_path: Path,
    image_path: Path,
    scene_prompt: str,
    verify_dir: Path,
) -> tuple[bool, str]:
    verify_dir.mkdir(parents=True, exist_ok=True)
    verify_output_path = verify_dir / "verify_output.png"
    if verify_output_path.exists():
        verify_output_path.unlink()

    command = [
        sys.executable,
        str(algorithm_path.resolve()),
        "--image-path",
        str(image_path.resolve()),
        "--output-path",
        str(verify_output_path.resolve()),
        "--scene-prompt",
        scene_prompt,
    ]
    context.log_event("tool_call", "contract_verify_start", {
                      "algorithm_path": algorithm_path, "verify_output": verify_output_path})
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        context.log_event("tool_call", "contract_verify_fail", {
                          "returncode": completed.returncode, "stderr": completed.stderr[:500]})
        return False, f"returncode={completed.returncode}, stderr={completed.stderr.strip()}"
    if not verify_output_path.exists():
        context.log_event("tool_call", "contract_verify_fail", {
                          "reason": "output file missing"})
        return False, "script exited 0 but did not create output_path"
    context.log_event("tool_call", "contract_verify_success",
                      {"output": verify_output_path})
    return True, "ok"


def research_stage(context: PipelineContext) -> ResearchResult:
    web_clues = _search_web_clues(context, context.request.scene_prompt)
    research_prompt = RESEARCH_PROMPT_TEMPLATE.format(
        scene_prompt=context.request.scene_prompt,
        web_context=_format_web_context(web_clues),
    )
    payload = _call_openrouter_json(context, research_prompt)
    payload_candidates = payload.get("candidates")
    if not isinstance(payload_candidates, list) or not payload_candidates:
        detail = {
            "error_code": "ResearchCandidatesInvalid",
            "message": "Research response has no valid candidates.",
            "payload_keys": sorted(payload.keys()),
            "candidate_count": len(payload_candidates) if isinstance(payload_candidates, list) else None,
            "stage": "research",
        }
        context.log_event("research", "invalid_candidates", detail)
        raise BusinessLogicError(json.dumps(detail, ensure_ascii=False))

    candidates: list[CandidateMethod] = []
    for item in payload_candidates:
        if not isinstance(item, dict):
            continue
        candidates.append(
            CandidateMethod(
                name=str(item.get("name") or "unknown_method"),
                description=str(item.get("description") or ""),
                parameters=item.get("parameters") if isinstance(
                    item.get("parameters"), dict) else {},
                rationale=str(item.get("rationale") or ""),
                sources=[str(source) for source in (
                    item.get("sources") or []) if str(source).strip()],
                confidence=float(item.get("confidence") or 0.0),
            )
        )
    if not candidates:
        detail = {
            "error_code": "ResearchCandidatesEmpty",
            "message": "Research response candidates are empty after parsing.",
            "candidate_count": 0,
            "stage": "research",
        }
        context.log_event("research", "empty_candidates_after_parse", detail)
        raise BusinessLogicError(json.dumps(detail, ensure_ascii=False))

    chosen_strategy = str(payload.get("chosen_strategy") or candidates[0].name)
    evaluation_metrics = [str(metric).strip() for metric in (
        payload.get("evaluation_metrics") or []) if str(metric).strip()]
    if not evaluation_metrics:
        evaluation_metrics = ["psnr", "ssim", "latency"]
    evaluation_plan = str(payload.get(
        "evaluation_plan") or "Compute objective metrics on input/output pair and aggregate score.")
    summary = str(payload.get("summary") or f"Selected {chosen_strategy}")
    sources = [str(source) for source in (
        payload.get("sources") or []) if str(source).strip()]
    if not sources and web_clues:
        sources = [clue.get("url", "")
                   for clue in web_clues if clue.get("url")]

    result = ResearchResult(
        scene_prompt=context.request.scene_prompt,
        candidates=candidates,
        chosen_strategy=chosen_strategy,
        evaluation_metrics=evaluation_metrics,
        evaluation_plan=evaluation_plan,
        summary=summary,
        sources=sources,
    )
    context.write_json("research.json", result.to_dict())
    context.write_json("research_web_clues.json", {"clues": web_clues})
    return result


def _generate_algorithm_source(research_result: ResearchResult) -> str:
    strategy = research_result.chosen_strategy
    parameters = research_result.candidates[0].parameters if research_result.candidates else {
    }
    serialized_parameters = json.dumps(
        parameters, ensure_ascii=False, indent=4)
    return f"""from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


STRATEGY_NAME = {strategy!r}
STRATEGY_PARAMETERS = {serialized_parameters}


def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    source_path = Path(image_path)
    destination_path = Path(output_path)
    image = Image.open(source_path).convert(\"RGB\")

    if STRATEGY_NAME == \"denoise_conservative\":
        processed = image.filter(ImageFilter.MedianFilter(size=int(STRATEGY_PARAMETERS.get(\"filter_size\", 3))))
        processed = ImageEnhance.Contrast(processed).enhance(float(STRATEGY_PARAMETERS.get(\"contrast\", 1.0)))
    elif STRATEGY_NAME == \"center_crop\":
        crop_ratio = float(STRATEGY_PARAMETERS.get(\"crop_ratio\", 0.5))
        crop_ratio = max(0.05, min(crop_ratio, 1.0))
        crop_width = max(1, int(image.width * crop_ratio))
        crop_height = max(1, int(image.height * crop_ratio))
        left = max(0, (image.width - crop_width) // 2)
        top = max(0, (image.height - crop_height) // 2)
        processed = image.crop((left, top, left + crop_width, top + crop_height))
    elif STRATEGY_NAME == \"deblur_sharpen\":
        processed = image.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
        processed = ImageEnhance.Sharpness(processed).enhance(float(STRATEGY_PARAMETERS.get(\"sharpness\", 1.0)))
    elif STRATEGY_NAME == \"upscale_refine\":
        scale = int(STRATEGY_PARAMETERS.get(\"scale\", 2))
        processed = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
        processed = ImageEnhance.Sharpness(processed).enhance(float(STRATEGY_PARAMETERS.get(\"sharpness\", 1.0)))
    elif STRATEGY_NAME == \"enhance_contrast\":
        processed = ImageOps.autocontrast(image)
        processed = ImageEnhance.Contrast(processed).enhance(float(STRATEGY_PARAMETERS.get(\"contrast\", 1.0)))
        processed = ImageEnhance.Sharpness(processed).enhance(float(STRATEGY_PARAMETERS.get(\"sharpness\", 1.0)))
    else:
        processed = ImageEnhance.Contrast(image).enhance(float(STRATEGY_PARAMETERS.get(\"contrast\", 1.0)))
        processed = ImageEnhance.Sharpness(processed).enhance(float(STRATEGY_PARAMETERS.get(\"sharpness\", 1.0)))

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(destination_path)
    return {{
        \"strategy_name\": STRATEGY_NAME,
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


def _validate_python_source(source: str) -> None:
    ast.parse(source)


def _validate_prepare_signature(source: str) -> None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names[:2] == ["input_image_path", "output_image_path"]:
                return
            raise ValueError(
                "prepare.py evaluate() must have arguments (input_image_path, output_image_path).")
    raise ValueError(
        "Generated prepare.py is missing evaluate(input_image_path, output_image_path).")


def codegen_stage(context: PipelineContext, research_result: ResearchResult) -> GeneratedAlgorithmArtifact:
    algo_dir = context.paths.generated_algorithms_dir
    algo_dir.mkdir(parents=True, exist_ok=True)
    strategy_fragment = _safe_filename_fragment(
        research_result.chosen_strategy)
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
    web_context = _format_web_context(web_clues_payload.get("clues") or [])
    prompt = CODEGEN_PROMPT_TEMPLATE.format(
        research_summary=research_summary, web_context=web_context)
    source = ""
    last_error = ""
    for attempt in range(3):
        context.log_event("codegen", "attempt", {"attempt": attempt + 1})
        source = _call_openrouter_code(context, prompt)
        _validate_python_source(source)
        _validate_run_signature(source)
        _validate_allowed_imports(source)
        algorithm_path.write_text(source, encoding="utf-8")
        ok, message = _verify_generated_algorithm_contract(
            context=context,
            algorithm_path=algorithm_path,
            image_path=context.artifacts["input_image"],
            scene_prompt=context.request.scene_prompt,
            verify_dir=context.paths.artifacts_dir / "codegen_verify",
        )
        if ok:
            break
        last_error = message
        prompt = (
            "Rewrite the COMPLETE Python file to satisfy contract.\n"
            f"Previous error: {message}\n"
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
        raise RuntimeError(
            f"Generated algorithm failed contract verification: {last_error}")

    prepare_prompt = PREPARE_CODEGEN_PROMPT_TEMPLATE.format(
        research_summary=research_summary, web_context=web_context)
    prepare_source = ""
    prepare_last_error = ""
    for attempt in range(3):
        context.log_event("codegen", "prepare_attempt",
                          {"attempt": attempt + 1})
        prepare_source = _call_openrouter_code(context, prepare_prompt)
        try:
            _validate_python_source(prepare_source)
            _validate_prepare_signature(prepare_source)
            _validate_allowed_imports(prepare_source)
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
        raise RuntimeError(
            f"Generated prepare.py failed validation: {prepare_last_error}")

    source_hash = sha256(source.encode("utf-8")).hexdigest()
    artifact = GeneratedAlgorithmArtifact(
        path=algorithm_path,
        prepare_path=prepare_path,
        source_hash=source_hash,
        strategy_name=research_result.chosen_strategy,
        syntax_validated=True,
    )
    context.artifacts["algorithm"] = algorithm_path
    context.artifacts["prepare"] = prepare_path
    context.write_json("codegen.json", artifact.to_dict())
    return artifact


class OptimizerAdapter:
    def optimize(
        self,
        algo_file_path: str,
        user_prompt: str,
        user_image_file_path: str,
        prepare_file_path: str | None = None,
    ) -> None:
        module_name = DEFAULT_OPTIMIZER_MODULE
        if not module_name:
            return

        module = __import__(module_name, fromlist=[DEFAULT_OPTIMIZER_FUNCTION])
        optimizer_function = getattr(module, DEFAULT_OPTIMIZER_FUNCTION)
        kwargs = {
            "algo_file_path": algo_file_path,
            "user_prompt": user_prompt,
            "user_image_file_path": user_image_file_path,
        }
        signature = inspect.signature(optimizer_function)
        if "prepare_file_path" in signature.parameters:
            kwargs["prepare_file_path"] = prepare_file_path
        optimizer_function(**kwargs)


def optimize_stage(context: PipelineContext, algorithm_artifact: GeneratedAlgorithmArtifact) -> tuple[bool, str]:
    before_text = algorithm_artifact.path.read_text(encoding="utf-8")
    before_hash = sha256(before_text.encode("utf-8")).hexdigest()
    context.log_event("tool_call", "optimizer_start", {
                      "algorithm_path": algorithm_artifact.path})
    adapter = OptimizerAdapter()
    adapter.optimize(
        algo_file_path=str(algorithm_artifact.path),
        user_prompt=context.request.scene_prompt,
        user_image_file_path=str(context.artifacts["input_image"]),
        prepare_file_path=str(algorithm_artifact.prepare_path.resolve(
        )) if algorithm_artifact.prepare_path else None,
    )
    after_text = algorithm_artifact.path.read_text(encoding="utf-8")
    after_hash = sha256(after_text.encode("utf-8")).hexdigest()
    if before_hash != after_hash:
        _validate_python_source(after_text)
        snapshot_path = context.paths.artifacts_dir / \
            f"{algorithm_artifact.path.stem}.optimized.py"
        shutil.copy2(algorithm_artifact.path, snapshot_path)
        context.artifacts["algorithm_optimized"] = snapshot_path
        context.log_event("tool_call", "optimizer_finish", {
                          "changed": True, "snapshot_path": snapshot_path})
        return True, "optimizer modified algorithm file in place"
    context.log_event("tool_call", "optimizer_finish", {"changed": False})
    return True, "optimizer completed without changing the algorithm file"


def execution_stage(context: PipelineContext, algorithm_artifact: GeneratedAlgorithmArtifact) -> ExecutionResult:
    output_image_path = context.paths.output_dir / \
        f"{context.run_id}_result{context.request.image_path.suffix or '.png'}"
    command = [
        sys.executable,
        str(algorithm_artifact.path.resolve()),
        "--image-path",
        str(context.artifacts["input_image"].resolve()),
        "--output-path",
        str(output_image_path.resolve()),
        "--scene-prompt",
        context.request.scene_prompt,
    ]
    start_time = time.time()
    context.log_event("tool_call", "executor_subprocess_start", {
                      "command": command})

    def _limit_resources() -> None:
        try:
            import resource

            if DEFAULT_MAX_MEMORY_MB:
                bytes_limit = DEFAULT_MAX_MEMORY_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS,
                                   (bytes_limit, bytes_limit))
        except Exception:
            return

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            cwd=str(context.paths.run_dir),
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )
        duration = time.time() - start_time
        success = completed.returncode == 0 and output_image_path.exists()
        result = ExecutionResult(
            success=success,
            command=command,
            returncode=completed.returncode,
            output_image_path=output_image_path if output_image_path.exists() else None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            timed_out=False,
        )
        context.write_json("execution.json", result.to_dict())
        context.log_event("tool_call", "executor_subprocess_finish", {
                          "duration_seconds": duration, "returncode": completed.returncode, "success": success})
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start_time
        result = ExecutionResult(
            success=False,
            command=command,
            returncode=None,
            output_image_path=output_image_path if output_image_path.exists() else None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_seconds=duration,
            timed_out=True,
        )
        context.write_json("execution.json", result.to_dict())
        context.log_event("tool_call", "executor_subprocess_timeout", {
                          "duration_seconds": duration})
        return result


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
    reference_gray = np.dot(reference[..., :3], [0.299, 0.587, 0.114])
    candidate_gray = np.dot(candidate[..., :3], [0.299, 0.587, 0.114])
    reference_gray = reference_gray.astype(np.float32)
    candidate_gray = candidate_gray.astype(np.float32)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_x = gaussian_filter(reference_gray, sigma=1.5)
    mu_y = gaussian_filter(candidate_gray, sigma=1.5)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = gaussian_filter(
        reference_gray * reference_gray, sigma=1.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(
        candidate_gray * candidate_gray, sigma=1.5) - mu_y_sq
    sigma_xy = gaussian_filter(
        reference_gray * candidate_gray, sigma=1.5) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    score = np.mean(numerator / (denominator + 1e-8))
    return float(np.clip(score, -1.0, 1.0))


def evaluate_stage(context: PipelineContext, execution_result: ExecutionResult) -> QualityReport:
    if not execution_result.output_image_path:
        raise FileNotFoundError("Execution did not produce an output image.")
    reference_image = Image.open(context.artifacts["input_image"])
    candidate_image = Image.open(execution_result.output_image_path)
    reference_array = _to_array(reference_image)
    candidate_array = _to_array(candidate_image, reference_image.size)
    psnr_value = _psnr(reference_array, candidate_array)
    ssim_value = _ssim(reference_array, candidate_array)
    latency_score = max(0.0, 1.0 - (execution_result.duration_seconds /
                        max(DEFAULT_TIMEOUT_SECONDS, 1)))
    normalized_psnr = min(psnr_value / 40.0, 1.0)
    score = float(
        np.mean([normalized_psnr, max(0.0, ssim_value), latency_score]))
    report = QualityReport(
        psnr=psnr_value,
        ssim=ssim_value,
        latency_seconds=execution_result.duration_seconds,
        score=score,
        notes="Higher score indicates better balance of quality and speed.",
    )
    context.write_json("quality.json", report.to_dict())
    return report


def package_stage(context: PipelineContext, payload: dict[str, Any]) -> Path:
    manifest_path = context.paths.manifest_path
    manifest_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
