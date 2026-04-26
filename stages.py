from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import ast
import importlib.util
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
from PIL import Image
from scipy.ndimage import gaussian_filter
from openai import OpenAI

from prompts import CODEGEN_PROMPT_TEMPLATE, PIPELINE_SYSTEM_PROMPT, RESEARCH_PROMPT_TEMPLATE

from .context import PipelineContext
from .models import (
    CandidateMethod,
    ExecutionResult,
    GeneratedAlgorithmArtifact,
    PipelineConfig,
    QualityReport,
    ResearchResult,
)


# ---------- 辅助函数 ----------
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


def _extract_python_code(text: str) -> str:
    text = text.strip()
    code_block = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if code_block:
        return code_block.group(1).strip() + "\n"
    return text + ("\n" if not text.endswith("\n") else "")


def _safe_filename_fragment(value: str) -> str:
    cleaned = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", value, flags=re.UNICODE).strip("_")
    return cleaned or "generated_algorithm"


def _create_openrouter_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for research/codegen.")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is required for research/codegen.")
    return OpenAI(api_key=api_key, base_url=base_url), model


def _call_openrouter_json(context: PipelineContext, prompt: str) -> dict[str, Any]:
    start_time = time.time()
    context.log_event("tool_call", "openrouter_json_start", {"prompt_preview": prompt[:240]})
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
    context.log_event("tool_call", "openrouter_json_finish", {
        "model": model, "duration_seconds": duration, "response_preview": content[:240]
    })
    return _extract_json_object(content)


def _call_openrouter_code(context: PipelineContext, prompt: str) -> str:
    start_time = time.time()
    context.log_event("tool_call", "openrouter_code_start", {"prompt_preview": prompt[:240]})
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
        "model": model, "duration_seconds": duration, "response_preview": content[:240]
    })
    return _extract_python_code(content)


def _search_web_clues(context: PipelineContext, scene_prompt: str) -> list[dict[str, str]]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        context.log_event("tool_call", "tavily_skip", {"reason": "TAVILY_API_KEY missing"})
        return []
    start_time = time.time()
    context.log_event("tool_call", "tavily_start", {"query": f"image processing algorithm for: {scene_prompt}"})
    try:
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
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        clues: list[dict[str, str]] = []
        for item in results[:5]:
            clues.append({
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "content": str(item.get("content") or "")[:500],
            })
        context.log_event("tool_call", "tavily_finish", {
            "duration_seconds": time.time() - start_time, "result_count": len(clues)
        })
        return clues
    except Exception as exc:
        context.log_event("tool_call", "tavily_error", {
            "duration_seconds": time.time() - start_time, "error": str(exc)
        })
        return []


def _format_web_context(clues: list[dict[str, str]]) -> str:
    if not clues:
        return "(no external web clues available)"
    lines: list[str] = []
    for idx, clue in enumerate(clues, 1):
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
            raise ValueError("run() must have arguments (image_path, output_path, scene_prompt).")
    raise ValueError("Generated algorithm is missing run(image_path, output_path, scene_prompt).")


def _validate_allowed_imports(source: str) -> None:
    allowed_third_party = {"numpy", "PIL", "scipy", "cv2", "skimage", "imageio"}
    stdlib_roots = set(getattr(sys, "stdlib_module_names", set()))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in stdlib_roots:
                    continue
                if root not in allowed_third_party:
                    raise ValueError(f"Disallowed third-party import: {alias.name}")
                if importlib.util.find_spec(root) is None:
                    raise ValueError(f"Unavailable import in environment: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            if root in stdlib_roots:
                continue
            if root not in allowed_third_party:
                raise ValueError(f"Disallowed third-party import: {node.module}")
            if importlib.util.find_spec(root) is None:
                raise ValueError(f"Unavailable import in environment: {node.module}")


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
        "algorithm_path": algorithm_path, "verify_output": verify_output_path
    })
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if completed.returncode != 0:
        context.log_event("tool_call", "contract_verify_fail", {
            "returncode": completed.returncode, "stderr": completed.stderr[:500]
        })
        return False, f"returncode={completed.returncode}, stderr={completed.stderr.strip()}"
    if not verify_output_path.exists():
        context.log_event("tool_call", "contract_verify_fail", {"reason": "output file missing"})
        return False, "script exited 0 but did not create output_path"
    context.log_event("tool_call", "contract_verify_success", {"output": verify_output_path})
    return True, "ok"


def _validate_python_source(source: str) -> None:
    ast.parse(source)


# ---------- 流水线阶段 ----------
def research_stage(context: PipelineContext) -> ResearchResult:
    web_clues = _search_web_clues(context, context.request.scene_prompt)
    research_prompt = RESEARCH_PROMPT_TEMPLATE.format(
        scene_prompt=context.request.scene_prompt,
        web_context=_format_web_context(web_clues),
    )
    payload = _call_openrouter_json(context, research_prompt)
    payload_candidates = payload.get("candidates")
    if not isinstance(payload_candidates, list) or not payload_candidates:
        raise ValueError("Research response has no valid candidates.")

    candidates: list[CandidateMethod] = []
    for item in payload_candidates:
        if not isinstance(item, dict):
            continue
        candidates.append(
            CandidateMethod(
                name=str(item.get("name") or "unknown_method"),
                description=str(item.get("description") or ""),
                parameters=item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
                rationale=str(item.get("rationale") or ""),
                sources=[str(source) for source in (item.get("sources") or []) if str(source).strip()],
                confidence=float(item.get("confidence") or 0.0),
            )
        )
    if not candidates:
        raise ValueError("Research response candidates are empty after parsing.")

    chosen_strategy = str(payload.get("chosen_strategy") or candidates[0].name)
    summary = str(payload.get("summary") or f"Selected {chosen_strategy}")
    sources = [str(source) for source in (payload.get("sources") or []) if str(source).strip()]
    if not sources and web_clues:
        sources = [clue.get("url", "") for clue in web_clues if clue.get("url")]

    result = ResearchResult(
        scene_prompt=context.request.scene_prompt,
        candidates=candidates,
        chosen_strategy=chosen_strategy,
        summary=summary,
        sources=sources,
    )
    # 不再写单独的 JSON，最终报告统一生成
    return result


def codegen_stage(context: PipelineContext, research_result: ResearchResult) -> GeneratedAlgorithmArtifact:
    algo_dir = context.paths.generated_dir
    strategy_fragment = _safe_filename_fragment(research_result.chosen_strategy)
    algorithm_path = algo_dir / f"algorithm_{strategy_fragment}.py"
    research_summary = json.dumps(research_result.to_dict(), ensure_ascii=False, indent=2)
    prompt = CODEGEN_PROMPT_TEMPLATE.format(research_summary=research_summary)
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
            verify_dir=context.paths.run_dir / ".verify",
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
            "Previous code:\n"
            f"```python\n{source}\n```"
        )
    else:
        raise RuntimeError(f"Generated algorithm failed contract verification: {last_error}")

    source_hash = sha256(source.encode("utf-8")).hexdigest()
    artifact = GeneratedAlgorithmArtifact(
        path=algorithm_path,
        source_hash=source_hash,
        strategy_name=research_result.chosen_strategy,
        syntax_validated=True,
    )
    context.artifacts["algorithm"] = algorithm_path
    return artifact


class OptimizerAdapter:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def optimize(self, algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None:
        module_name = self.config.optimizer_module
        if not module_name:
            return
        module = __import__(module_name, fromlist=[self.config.optimizer_function])
        optimizer_function = getattr(module, self.config.optimizer_function)
        optimizer_function(
            algo_file_path=algo_file_path,
            user_prompt=user_prompt,
            user_image_file_path=user_image_file_path,
        )


def optimize_stage(context: PipelineContext, algorithm_artifact: GeneratedAlgorithmArtifact) -> tuple[bool, str]:
    before_text = algorithm_artifact.path.read_text(encoding="utf-8")
    before_hash = sha256(before_text.encode("utf-8")).hexdigest()
    context.log_event("tool_call", "optimizer_start", {"algorithm_path": algorithm_artifact.path})
    adapter = OptimizerAdapter(context.request.config)
    adapter.optimize(
        algo_file_path=str(algorithm_artifact.path),
        user_prompt=context.request.scene_prompt,
        user_image_file_path=str(context.artifacts["input_image"]),
    )
    after_text = algorithm_artifact.path.read_text(encoding="utf-8")
    after_hash = sha256(after_text.encode("utf-8")).hexdigest()
    changed = before_hash != after_hash
    if changed:
        _validate_python_source(after_text)
    context.log_event("tool_call", "optimizer_finish", {"changed": changed})
    msg = "optimizer modified algorithm file" if changed else "optimizer completed without changes"
    return True, msg


def execution_stage(context: PipelineContext, algorithm_artifact: GeneratedAlgorithmArtifact) -> ExecutionResult:
    # 输出图片命名更友好：原文件名_processed.后缀
    orig_name = context.request.image_path.stem
    suffix = context.request.image_path.suffix or ".jpg"
    output_image_path = context.paths.result_dir / f"{orig_name}_processed{suffix}"

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
    context.log_event("tool_call", "executor_subprocess_start", {"command": command})

    def _limit_resources() -> None:
        try:
            import resource
            if context.request.config.max_memory_mb:
                bytes_limit = context.request.config.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except Exception:
            return

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=context.request.config.timeout_seconds,
            cwd=str(context.paths.run_dir),
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )
        duration = time.time() - start_time
        success = completed.returncode == 0 and output_image_path.exists()
        result = ExecutionResult(
            success=success,
            command=command,
            returncode=completed.returncode,
            output_image_path=output_image_path if success else None,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            timed_out=False,
        )
        context.log_event("tool_call", "executor_subprocess_finish", {
            "duration_seconds": duration, "returncode": completed.returncode, "success": success
        })
        return result
    except subprocess.TimeoutExpired as exc:
        duration = time.time() - start_time
        result = ExecutionResult(
            success=False,
            command=command,
            returncode=None,
            output_image_path=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            duration_seconds=duration,
            timed_out=True,
        )
        context.log_event("tool_call", "executor_subprocess_timeout", {"duration_seconds": duration})
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
    sigma_x_sq = gaussian_filter(reference_gray * reference_gray, sigma=1.5) - mu_x_sq
    sigma_y_sq = gaussian_filter(candidate_gray * candidate_gray, sigma=1.5) - mu_y_sq
    sigma_xy = gaussian_filter(reference_gray * candidate_gray, sigma=1.5) - mu_xy
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
    latency_score = max(0.0, 1.0 - (execution_result.duration_seconds / max(context.request.config.timeout_seconds, 1)))
    normalized_psnr = min(psnr_value / 40.0, 1.0)
    score = float(np.mean([normalized_psnr, max(0.0, ssim_value), latency_score]))
    report = QualityReport(
        psnr=psnr_value,
        ssim=ssim_value,
        latency_seconds=execution_result.duration_seconds,
        score=score,
        notes="Higher score indicates better balance of quality and speed.",
    )
    return report


def package_stage(context: PipelineContext, payload: dict[str, Any]) -> Path:
    """写入最终报告并清理临时文件"""
    report_path = context.paths.report_path
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 清理临时目录
    temp_dir = context.paths.run_dir / ".temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    verify_dir = context.paths.run_dir / ".verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    return report_path