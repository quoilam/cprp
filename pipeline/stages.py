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
    CODEGEN_PROMPT_TEMPLATE_FOR_CANDIDATE,
    PIPELINE_SYSTEM_PROMPT,
    PREPARE_CODEGEN_PROMPT_TEMPLATE,
    RESEARCH_PROMPT_TEMPLATE,
)

from .context import PipelineContext
from .models import (
    CandidateMethod,
    CodegenBranchResult,
    OptimizedArtifact,
    ExecutionBranchResult,
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


def _coerce_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                parts.append(text_attr)
        return "\n".join(part for part in parts if part)
    return str(value)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        choices = completion.get("choices") or []
        if choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                return _coerce_message_content(message.get("content"))
        return _coerce_message_content(completion.get("content"))

    choices = getattr(completion, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return _coerce_message_content(getattr(message, "content", None))
    return _coerce_message_content(getattr(completion, "content", None))


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
    if base_url.rstrip("/").endswith("/api"):
        base_url = base_url.rstrip("/") + "/v1"
    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is required for research/codegen.")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _looks_like_html_response(text: str) -> bool:
    head = text.lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


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
        content = _extract_completion_text(completion)
        if _looks_like_html_response(content):
            raise BusinessLogicError(
                "OpenRouter returned HTML instead of JSON. Check OPENROUTER_BASE_URL and model configuration."
            )
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
    policy = build_retry_policy(
        max_retries=DEFAULT_LLM_MAX_RETRIES,
        initial_delay=DEFAULT_RETRY_INITIAL_DELAY,
        max_delay=DEFAULT_RETRY_MAX_DELAY,
        jitter=DEFAULT_RETRY_JITTER,
    )
    llm_timeout = float(DEFAULT_LLM_TIMEOUT_SECONDS)
    context.log_event("tool_call", "openrouter_code_start",
                      {
                          "prompt_preview": prompt[:240],
                          "timeout_seconds": llm_timeout,
                          "max_retries": policy.max_retries,
                          "max_attempts": policy.max_attempts,
                      })
    client, model = _create_openrouter_client()

    @build_retry_decorator(
        context=context,
        stage="codegen",
        operation="openrouter_code",
        policy=policy,
    )
    def _call_once() -> tuple[str, str]:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": PIPELINE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            timeout=llm_timeout,
        )
        content = _extract_completion_text(completion)
        if _looks_like_html_response(content):
            raise BusinessLogicError(
                "OpenRouter returned HTML instead of code. Check OPENROUTER_BASE_URL and model configuration."
            )
        source = _extract_python_code(content)
        if not source.strip():
            raise ModelResponseParseError(
                "OpenRouter code response is empty.",
                retryable=True,
            )
        return content, source

    content, source = _call_once()
    duration = time.time() - start_time
    context.log_event("tool_call", "openrouter_code_finish", {
                      "model": model, "duration_seconds": duration, "response_preview": content[:240]})
    return source


def _search_web_clues(context: PipelineContext, scene_prompt: str) -> list[dict[str, str]]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        context.log_event("tool_call", "tavily_skip", {
                          "reason": "TAVILY_API_KEY missing"})
        return []
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


def _build_local_research_result(scene_prompt: str, web_clues: list[dict[str, str]]) -> ResearchResult:
    """Build a deterministic research result without external LLM calls."""
    primary_strategy, primary_params, primary_rationale = _choose_strategy(scene_prompt)

    secondary_strategy = "enhance_contrast"
    secondary_params: dict[str, Any] = {"contrast": 1.12, "sharpness": 1.06}
    if primary_strategy == secondary_strategy:
        secondary_strategy = "denoise_conservative"
        secondary_params = {"filter_size": 3, "contrast": 1.03}

    candidates = [
        CandidateMethod(
            name=primary_strategy,
            description=f"Deterministic baseline selected from prompt intent: {primary_rationale}",
            parameters=primary_params,
            rationale=primary_rationale,
            sources=["local_heuristic"],
            confidence=0.82,
        ),
        CandidateMethod(
            name=secondary_strategy,
            description="Secondary robust fallback branch for multi-branch execution stability.",
            parameters=secondary_params,
            rationale="Provide diversity to increase chance of at least one successful branch.",
            sources=["local_heuristic"],
            confidence=0.68,
        ),
    ]

    sources = [str(clue.get("url") or "") for clue in web_clues if str(clue.get("url") or "").strip()]

    return ResearchResult(
        scene_prompt=scene_prompt,
        candidates=candidates,
        chosen_strategy=candidates[0].name,
        evaluation_metrics=["psnr", "ssim", "latency"],
        evaluation_plan="Compute objective metrics on input/output pair and aggregate score.",
        summary="Built deterministic local research result with two candidate branches.",
        sources=sources,
    )


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


def research_stage(context: PipelineContext, bypass_autoresearch: bool = False) -> ResearchResult:
    web_clues = _search_web_clues(context, context.request.scene_prompt)
    if bypass_autoresearch:
        context.log_event("research", "bypass_autoresearch", {
                          "enabled": True,
                          "reason": "Skip external LLM research and use deterministic local candidates."})
        result = _build_local_research_result(context.request.scene_prompt, web_clues)
        context.write_json("research.json", result.to_dict())
        context.write_json("research_web_clues.json", {"clues": web_clues})
        return result

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


def _generate_algorithm_source_with_strategy(strategy: str, parameters: dict[str, Any]) -> str:
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


def _generate_algorithm_source(research_result: ResearchResult) -> str:
    strategy = research_result.chosen_strategy
    parameters = research_result.candidates[0].parameters if research_result.candidates else {}
    return _generate_algorithm_source_with_strategy(strategy, parameters)


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
        context.log_event("codegen", "fallback_local_generator", {
                          "reason": last_error[:200], "strategy": research_result.chosen_strategy})
        source = _generate_algorithm_source(research_result)
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
        if not ok:
            raise RuntimeError(
                f"Generated algorithm failed contract verification: {last_error}; local fallback failed: {message}")

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


# =============================================================================
# Multi-branch pipeline functions
# =============================================================================

def codegen_branch(
    context: PipelineContext,
    candidate: CandidateMethod,
    candidate_index: int,
    research_result: ResearchResult,
    web_context: str,
) -> GeneratedAlgorithmArtifact | None:
    """Generate code for a single candidate branch. Returns None on failure."""
    algo_dir = context.paths.generated_algorithms_dir
    algo_dir.mkdir(parents=True, exist_ok=True)
    strategy_fragment = _safe_filename_fragment(candidate.name)
    algorithm_path = algo_dir / f"{context.run_id}_{strategy_fragment}.py"
    prepare_path = algo_dir / f"{context.run_id}_{strategy_fragment}.prepare.py"

    research_summary = json.dumps(
        research_result.to_dict(), ensure_ascii=False, indent=2)
    prompt = CODEGEN_PROMPT_TEMPLATE_FOR_CANDIDATE.format(
        research_summary=research_summary,
        candidate_index=candidate_index + 1,
        candidate_name=candidate.name,
        candidate_description=candidate.description,
        candidate_parameters=json.dumps(candidate.parameters, ensure_ascii=False, indent=4),
        candidate_rationale=candidate.rationale,
        candidate_confidence=candidate.confidence,
        web_context=web_context,
    )

    source = ""
    last_error = ""
    for attempt in range(3):
        context.log_event("codegen_branch", "attempt", {
                          "candidate_index": candidate_index, "attempt": attempt + 1, "candidate_name": candidate.name})
        try:
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
                verify_dir=context.paths.artifacts_dir / f"codegen_verify_{candidate_index}",
            )
            if ok:
                context.log_event("codegen_branch", "success", {
                                  "candidate_index": candidate_index, "path": str(algorithm_path)})
                break
            last_error = message
        except Exception as exc:
            last_error = str(exc)
            context.log_event("codegen_branch", "attempt_error", {
                              "candidate_index": candidate_index, "attempt": attempt + 1, "error": last_error[:200]})
        # Build retry prompt
        prompt = (
            "Rewrite the COMPLETE Python file to satisfy contract.\n"
            f"Previous error: {last_error}\n"
            "STRICT constraints:\n"
            "- Allowed libraries: Python stdlib, Pillow(PIL), numpy, scipy, cv2, skimage, imageio.\n"
            "- Must define run(image_path, output_path, scene_prompt) and save image to output_path.\n"
            "- Keep a runnable CLI main with --image-path --output-path --scene-prompt.\n"
            "- Output Python code only.\n"
            f"Target algorithm: {candidate.name} - {candidate.description}\n"
            f"Parameters: {json.dumps(candidate.parameters, ensure_ascii=False)}\n"
            "Research summary:\n"
            f"{research_summary}\n"
            "Web clues:\n"
            f"{web_context}\n"
            "Previous code:\n"
            f"```python\n{source}\n```"
        )
    else:
        context.log_event("codegen_branch", "fallback_local_generator", {
                          "candidate_index": candidate_index, "last_error": last_error[:200], "strategy": candidate.name})
        try:
            source = _generate_algorithm_source_with_strategy(candidate.name, candidate.parameters)
            _validate_python_source(source)
            _validate_run_signature(source)
            _validate_allowed_imports(source)
            algorithm_path.write_text(source, encoding="utf-8")
            ok, message = _verify_generated_algorithm_contract(
                context=context,
                algorithm_path=algorithm_path,
                image_path=context.artifacts["input_image"],
                scene_prompt=context.request.scene_prompt,
                verify_dir=context.paths.artifacts_dir / f"codegen_verify_{candidate_index}",
            )
            if not ok:
                context.log_event("codegen_branch", "failed", {
                                  "candidate_index": candidate_index, "last_error": f"local fallback failed: {message}"[:200]})
                return None
        except Exception as exc:
            context.log_event("codegen_branch", "failed", {
                              "candidate_index": candidate_index, "last_error": str(exc)[:200]})
            return None

    # Generate prepare file
    prepare_prompt = PREPARE_CODEGEN_PROMPT_TEMPLATE.format(
        research_summary=research_summary, web_context=web_context)
    prepare_source = ""
    prepare_last_error = ""
    for attempt in range(3):
        try:
            prepare_source = _call_openrouter_code(context, prepare_prompt)
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
                f"Research summary:\n{research_summary}\n"
                f"Web clues:\n{web_context}\n"
                f"Previous code:\n```python\n{prepare_source}\n```"
            )
    else:
        context.log_event("codegen_branch", "prepare_failed", {
                          "candidate_index": candidate_index, "last_error": prepare_last_error[:200]})
        # Even if prepare fails, we still have a valid algorithm - log and continue

    source_hash = sha256(source.encode("utf-8")).hexdigest()
    artifact = GeneratedAlgorithmArtifact(
        path=algorithm_path,
        prepare_path=prepare_path if prepare_path.exists() else None,
        source_hash=source_hash,
        strategy_name=candidate.name,
        syntax_validated=True,
        candidate_index=candidate_index,
    )
    return artifact


def codegen_multi_stage(
    context: PipelineContext,
    research_result: ResearchResult,
    max_branches: int = 2,
) -> list[CodegenBranchResult]:
    """Generate code for multiple candidate methods (multi-branch)."""
    # Load web clues
    web_clues_path = context.paths.run_dir / "research_web_clues.json"
    web_clues_payload = {}
    if web_clues_path.exists():
        web_clues_payload = json.loads(
            web_clues_path.read_text(encoding="utf-8"))
    web_context = _format_web_context(web_clues_payload.get("clues") or [])

    # Limit to max_branches candidates (sorted by confidence descending)
    sorted_candidates = sorted(
        research_result.candidates,
        key=lambda c: c.confidence,
        reverse=True,
    )[:max_branches]

    context.log_event("codegen_multi", "start", {
                      "total_candidates": len(research_result.candidates),
                      "branches_to_generate": len(sorted_candidates),
                      "candidates": [c.name for c in sorted_candidates]})

    branches: list[CodegenBranchResult] = []
    for idx, candidate in enumerate(sorted_candidates):
        context.log_event("codegen_multi", "branch_start", {
                          "branch_index": idx, "candidate_name": candidate.name, "confidence": candidate.confidence})
        artifact = codegen_branch(
            context=context,
            candidate=candidate,
            candidate_index=idx,
            research_result=research_result,
            web_context=web_context,
        )
        if artifact is not None:
            branches.append(CodegenBranchResult(
                candidate=candidate,
                artifact=artifact,
            ))
            context.log_event("codegen_multi", "branch_success", {
                              "branch_index": idx, "artifact_path": str(artifact.path)})
        else:
            context.log_event("codegen_multi", "branch_failure", {
                              "branch_index": idx, "candidate_name": candidate.name})

    # Write multi-branch manifest
    context.write_json("codegen_multi.json", {
        "branches": [b.to_dict() for b in branches],
        "total_generated": len(branches),
        "total_requested": len(sorted_candidates),
    })

    context.log_event("codegen_multi", "finish", {
                      "branches_generated": len(branches),
                      "branch_names": [b.candidate.name for b in branches]})
    return branches


def optimize_branch_with_bypass(
    context: PipelineContext,
    artifact: GeneratedAlgorithmArtifact,
    bypass: bool = True,
) -> OptimizedArtifact:
    """Optimize a single branch. If bypass=True, skip autoresearch and just snapshot.

    The autoresearch agent takes a very long time, so we offer a bypass mode that:
    1. Validates the existing code
    2. Creates a snapshot
    3. Marks as 'optimized' (with local optimization if needed)
    """
    before_text = artifact.path.read_text(encoding="utf-8")
    before_hash = sha256(before_text.encode("utf-8")).hexdigest()
    candidate_index = artifact.candidate_index

    context.log_event("optimize_branch", "start", {
                      "candidate_index": candidate_index,
                      "algorithm_path": str(artifact.path),
                      "bypass": bypass})

    if bypass:
        # Bypass autoresearch - just validate and snapshot
        context.log_event("optimize_branch", "bypass_autoresearch", {
                          "candidate_index": candidate_index,
                          "reason": "Skipping time-consuming autoresearch"})
        # The code is already generated and validated, so we just snapshot
        snapshot_path = artifact.path.parent / f"{artifact.path.stem}.optimized.py"
        shutil.copy2(artifact.path, snapshot_path)
        after_hash = before_hash  # No change in bypass mode
        context.log_event("optimize_branch", "bypass_snapshot", {
                          "candidate_index": candidate_index,
                          "snapshot_path": str(snapshot_path)})
    else:
        # Call the actual optimizer (autoresearch)
        try:
            adapter = OptimizerAdapter()
            adapter.optimize(
                algo_file_path=str(artifact.path),
                user_prompt=context.request.scene_prompt,
                user_image_file_path=str(context.artifacts["input_image"]),
                prepare_file_path=str(artifact.prepare_path.resolve(
                )) if artifact.prepare_path else None,
            )
        except Exception as exc:
            context.log_event("optimize_branch", "optimizer_error", {
                              "candidate_index": candidate_index, "error": str(exc)[:200]})
            # On optimizer failure, fallback to bypass
            context.log_event("optimize_branch", "fallback_to_bypass", {
                              "candidate_index": candidate_index})

        after_text = artifact.path.read_text(encoding="utf-8")
        after_hash = sha256(after_text.encode("utf-8")).hexdigest()

        if before_hash != after_hash:
            _validate_python_source(after_text)
            snapshot_path = artifact.path.parent / f"{artifact.path.stem}.optimized.py"
            shutil.copy2(artifact.path, snapshot_path)
        else:
            snapshot_path = artifact.path

    changed = (before_hash != after_hash)

    optimized = OptimizedArtifact(
        original_path=artifact.path,
        optimized_path=artifact.path if changed else snapshot_path,
        strategy_name=artifact.strategy_name,
        candidate_index=candidate_index,
        changed=changed,
        snapshot_path=snapshot_path,
        before_hash=before_hash,
        after_hash=after_hash,
    )

    context.log_event("optimize_branch", "finish", {
                      "candidate_index": candidate_index,
                      "changed": changed,
                      "snapshot_path": str(snapshot_path)})
    return optimized


def optimize_multi_stage(
    context: PipelineContext,
    branches: list[CodegenBranchResult],
    bypass_autoresearch: bool = True,
) -> list[OptimizedArtifact]:
    """Optimize all branches. Supports bypass mode for autoresearch."""
    context.log_event("optimize_multi", "start", {
                      "total_branches": len(branches),
                      "bypass_autoresearch": bypass_autoresearch})

    optimized_results: list[OptimizedArtifact] = []
    for branch in branches:
        result = optimize_branch_with_bypass(
            context=context,
            artifact=branch.artifact,
            bypass=bypass_autoresearch,
        )
        optimized_results.append(result)

    context.write_json("optimize_multi.json", {
        "branches": [r.to_dict() for r in optimized_results],
        "bypass_autoresearch": bypass_autoresearch,
    })

    context.log_event("optimize_multi", "finish", {
                      "optimized_branches": len(optimized_results)})
    return optimized_results


def execution_branch(
    context: PipelineContext,
    optimized: OptimizedArtifact,
    branch_index: int,
) -> ExecutionBranchResult:
    """Execute a single optimized branch."""
    algorithm_path = optimized.optimized_path
    original_suffix = context.request.image_path.suffix
    if original_suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        original_suffix = ".png"
    output_image_path = context.paths.output_dir / \
        f"{context.run_id}_result_{branch_index}{original_suffix or '.png'}"
    command = [
        sys.executable,
        str(algorithm_path.resolve()),
        "--image-path",
        str(context.artifacts["input_image"].resolve()),
        "--output-path",
        str(output_image_path.resolve()),
        "--scene-prompt",
        context.request.scene_prompt,
    ]

    start_time = time.time()
    context.log_event("execution_branch", "start", {
                      "branch_index": branch_index, "command": command,
                      "algorithm_path": str(algorithm_path)})

    def _limit_resources() -> None:
        try:
            import resource
            if DEFAULT_MAX_MEMORY_MB:
                bytes_limit = DEFAULT_MAX_MEMORY_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
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
        exec_result = ExecutionResult(
            success=success,
            command=command,
            returncode=completed.returncode,
            output_image_path=output_image_path if output_image_path.exists() else None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            timed_out=False,
        )
        context.log_event("execution_branch", "finish", {
                          "branch_index": branch_index, "success": success,
                          "duration_seconds": duration, "returncode": completed.returncode})
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start_time
        exec_result = ExecutionResult(
            success=False,
            command=command,
            returncode=None,
            output_image_path=output_image_path if output_image_path.exists() else None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_seconds=duration,
            timed_out=True,
        )
        context.log_event("execution_branch", "timeout", {
                          "branch_index": branch_index, "duration_seconds": duration})

    return ExecutionBranchResult(
        execution_result=exec_result,
        strategy_name=optimized.strategy_name,
        candidate_index=branch_index,
        optimized=True,
    )


def execute_multi_stage(
    context: PipelineContext,
    optimized_branches: list[OptimizedArtifact],
) -> list[ExecutionBranchResult]:
    """Execute all optimized branches."""
    context.log_event("execute_multi", "start", {
                      "total_branches": len(optimized_branches)})

    exec_results: list[ExecutionBranchResult] = []
    for optimized in optimized_branches:
        result = execution_branch(
            context=context,
            optimized=optimized,
            branch_index=optimized.candidate_index,
        )
        exec_results.append(result)

    # Write per-branch execution results
    for result in exec_results:
        context.write_json(
            f"execution_branch_{result.candidate_index}.json",
            result.to_dict(),
        )

    context.write_json("execute_multi.json", {
        "branches": [r.to_dict() for r in exec_results],
        "successful_branches": sum(1 for r in exec_results if r.execution_result.success),
    })

    context.log_event("execute_multi", "finish", {
                      "total_executed": len(exec_results),
                      "successful": sum(1 for r in exec_results if r.execution_result.success)})
    return exec_results


def evaluate_branch(
    context: PipelineContext,
    exec_branch_result: ExecutionBranchResult,
) -> QualityReport | None:
    """Evaluate a single branch's execution result."""
    exec_result = exec_branch_result.execution_result
    if not exec_result.output_image_path:
        context.log_event("evaluate_branch", "skip_no_output", {
                          "branch_index": exec_branch_result.candidate_index})
        return None

    branch_index = exec_branch_result.candidate_index
    verify_dir = context.paths.artifacts_dir / f"evaluate_{branch_index}"
    verify_dir.mkdir(parents=True, exist_ok=True)

    reference_image = Image.open(context.artifacts["input_image"])
    candidate_image = Image.open(exec_result.output_image_path)
    reference_array = _to_array(reference_image)
    candidate_array = _to_array(candidate_image, reference_image.size)

    psnr_value = _psnr(reference_array, candidate_array)
    ssim_value = _ssim(reference_array, candidate_array)
    latency_score = max(0.0, 1.0 - (exec_result.duration_seconds /
                        max(DEFAULT_TIMEOUT_SECONDS, 1)))
    normalized_psnr = min(psnr_value / 40.0, 1.0)
    score = float(np.mean([normalized_psnr, max(0.0, ssim_value), latency_score]))

    report = QualityReport(
        psnr=psnr_value,
        ssim=ssim_value,
        latency_seconds=exec_result.duration_seconds,
        score=score,
        notes=f"Branch {branch_index}: {exec_branch_result.strategy_name}",
    )

    context.write_json(
        f"quality_branch_{branch_index}.json",
        report.to_dict(),
    )

    context.log_event("evaluate_branch", "success", {
                      "branch_index": branch_index,
                      "psnr": psnr_value,
                      "ssim": ssim_value,
                      "score": score})
    return report


def evaluate_multi_stage(
    context: PipelineContext,
    exec_branches: list[ExecutionBranchResult],
) -> list[tuple[int, QualityReport]]:
    """Evaluate all branches and return quality reports."""
    context.log_event("evaluate_multi", "start", {
                      "total_branches": len(exec_branches)})

    quality_results: list[tuple[int, QualityReport]] = []
    for exec_result in exec_branches:
        report = evaluate_branch(context, exec_result)
        if report is not None:
            quality_results.append((exec_result.candidate_index, report))

    # Write comparison report
    context.write_json("evaluate_multi.json", {
        "branches": [
            {"branch_index": idx, "quality_report": report.to_dict()}
            for idx, report in quality_results
        ],
        "total_evaluated": len(quality_results),
    })

    context.log_event("evaluate_multi", "finish", {
                      "total_evaluated": len(quality_results)})
    return quality_results


def select_best_branch(
    context: PipelineContext,
    exec_branches: list[ExecutionBranchResult],
    quality_branches: list[tuple[int, QualityReport]],
    selection_strategy: str = "best_score",
) -> tuple[int, ExecutionBranchResult, QualityReport] | None:
    """Select the best branch based on the given strategy.

    Strategies:
    - "best_score": Select by highest quality composite score
    - "highest_confidence": Select by candidate confidence (from research)
    - "first_success": Select the first successful branch
    """
    if not quality_branches:
        context.log_event("select_best", "no_branches", {"reason": "No quality results"})
        return None

    context.log_event("select_best", "start", {
                      "strategy": selection_strategy,
                      "total_branches": len(quality_branches)})

    selected: tuple[int, ExecutionBranchResult, QualityReport] | None = None

    if selection_strategy == "best_score":
        # Select the branch with highest composite quality score
        selected_idx = max(quality_branches, key=lambda x: x[1].score)[0]
        for idx, report in quality_branches:
            exec_result = next(
                (e for e in exec_branches if e.candidate_index == idx), None)
            if exec_result and idx == selected_idx:
                selected = (idx, exec_result, report)
                break

    elif selection_strategy == "highest_confidence":
        # Select the branch with highest confidence from research
        # This requires matching index to candidates - use first successful
        for idx, report in quality_branches:
            exec_result = next(
                (e for e in exec_branches if e.candidate_index == idx), None)
            if exec_result and exec_result.execution_result.success:
                selected = (idx, exec_result, report)
                break

    elif selection_strategy == "first_success":
        # Select the first successful branch
        for idx, report in quality_branches:
            exec_result = next(
                (e for e in exec_branches if e.candidate_index == idx), None)
            if exec_result and exec_result.execution_result.success:
                selected = (idx, exec_result, report)
                break

    if selected is None:
        context.log_event("select_best", "no_success_branch", {})
        return None

    idx, exec_result, report = selected
    context.log_event("select_best", "selected", {
                      "branch_index": idx,
                      "strategy": selection_strategy,
                      "quality_score": report.score,
                      "strategy_name": exec_result.strategy_name})

    return selected
