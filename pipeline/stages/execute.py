from __future__ import annotations

import os
import subprocess
import sys
import time

from pipeline.context import PipelineContext
from pipeline.models import (
    ExecutionResult,
    GeneratedAlgorithmArtifact,
)


def _safe_output_suffix(suffix: str) -> str:
    normalized = (suffix or "").lower()
    if normalized in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        return suffix
    return ".png"


def _limit_resources(max_memory_mb: int | None) -> None:
    try:
        import resource

        if max_memory_mb:
            bytes_limit = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    except Exception:
        return


def execution_stage(context: PipelineContext, algorithm_artifact: GeneratedAlgorithmArtifact) -> ExecutionResult:
    suffix = _safe_output_suffix(context.request.image_path.suffix)
    output_image_path = context.paths.output_dir / \
        f"{context.run_id}_result{suffix}"
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
    timeout_seconds = context.config.executor_timeout_seconds
    max_memory_mb = context.config.max_memory_mb
    context.log_event("tool_call", "executor_subprocess_start", {
                      "command": command})

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(context.paths.run_dir),
            preexec_fn=(lambda: _limit_resources(max_memory_mb)) if os.name == "posix" else None,
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
        context.log_event(
            "tool_call",
            "executor_subprocess_finish",
            {"duration_seconds": duration,
                "returncode": completed.returncode, "success": success},
        )
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
