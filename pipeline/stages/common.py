from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from prompts import PIPELINE_SYSTEM_PROMPT
from pipeline.context import PipelineContext
from pipeline.resilience import (
    BusinessLogicError,
    ModelResponseParseError,
    build_retry_decorator,
    build_retry_policy,
)


def safe_filename_fragment(value: str) -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_",
                     value, flags=re.UNICODE).strip("_")
    return cleaned or "generated_algorithm"


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
            message = choices[0].get("message") if isinstance(
                choices[0], dict) else None
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


def _build_retry_policy(context: PipelineContext, *, max_retries: int | None = None):
    config = context.config
    return build_retry_policy(
        max_retries=config.llm_max_retries if max_retries is None else max_retries,
        initial_delay=config.retry_initial_delay,
        max_delay=config.retry_max_delay,
        jitter=config.retry_jitter,
    )


def _looks_like_html_response(text: str) -> bool:
    head = text.lstrip().lower()
    return head.startswith("<!doctype html") or head.startswith("<html")


def call_openrouter_json(context: PipelineContext, prompt: str) -> dict[str, Any]:
    start_time = time.time()
    policy = _build_retry_policy(context)
    llm_timeout = float(context.config.llm_timeout_seconds)
    context.log_event(
        "tool_call",
        "openrouter_json_start",
        {
            "prompt_preview": prompt[:240],
            "timeout_seconds": llm_timeout,
            "max_retries": policy.max_retries,
            "max_attempts": policy.max_attempts,
        },
    )
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
                message, retryable=is_truncated) from exc

    content, payload = _call_once()
    duration = time.time() - start_time
    context.log_event(
        "tool_call",
        "openrouter_json_finish",
        {
            "model": model,
            "duration_seconds": duration,
            "response_preview": content[:240],
        },
    )
    return payload


def call_openrouter_code(context: PipelineContext, prompt: str) -> str:
    start_time = time.time()
    policy = _build_retry_policy(context)
    llm_timeout = float(context.config.llm_timeout_seconds)
    context.log_event(
        "tool_call",
        "openrouter_code_start",
        {
            "prompt_preview": prompt[:240],
            "timeout_seconds": llm_timeout,
            "max_retries": policy.max_retries,
            "max_attempts": policy.max_attempts,
        },
    )
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
                "OpenRouter code response is empty.", retryable=True)
        return content, source

    content, source = _call_once()
    duration = time.time() - start_time
    context.log_event(
        "tool_call",
        "openrouter_code_finish",
        {
            "model": model,
            "duration_seconds": duration,
            "response_preview": content[:240],
        },
    )
    return source


def validate_python_source(source: str) -> None:
    ast.parse(source)


def validate_run_signature(source: str) -> None:
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


def validate_prepare_signature(source: str) -> None:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            arg_names = [arg.arg for arg in node.args.args]
            if arg_names[:2] == ["input_image_path", "output_image_path"]:
                return
            raise ValueError(
                "prepare.py evaluate() must have arguments (input_image_path, output_image_path)."
            )
    raise ValueError(
        "Generated prepare.py is missing evaluate(input_image_path, output_image_path).")


def validate_allowed_imports(source: str) -> None:
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


def verify_generated_algorithm_contract(
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
    context.log_event(
        "tool_call",
        "contract_verify_start",
        {"algorithm_path": algorithm_path, "verify_output": verify_output_path},
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        context.log_event(
            "tool_call",
            "contract_verify_fail",
            {"returncode": completed.returncode,
                "stderr": completed.stderr[:500]},
        )
        return False, f"returncode={completed.returncode}, stderr={completed.stderr.strip()}"
    if not verify_output_path.exists():
        context.log_event("tool_call", "contract_verify_fail", {
                          "reason": "output file missing"})
        return False, "script exited 0 but did not create output_path"
    context.log_event("tool_call", "contract_verify_success",
                      {"output": verify_output_path})
    return True, "ok"
