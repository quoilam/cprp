from __future__ import annotations

import inspect
import shutil
from hashlib import sha256

from pipeline.context import PipelineContext
from pipeline.models import (
    DEFAULT_OPTIMIZER_FUNCTION,
    DEFAULT_OPTIMIZER_MODULE,
    GeneratedAlgorithmArtifact,
)

from .common import validate_python_source


class OptimizerAdapter:
    def __init__(self, module_name: str | None = None, function_name: str | None = None):
        self.module_name = module_name or DEFAULT_OPTIMIZER_MODULE
        self.function_name = function_name or DEFAULT_OPTIMIZER_FUNCTION

    def optimize(
        self,
        algo_file_path: str,
        user_prompt: str,
        user_image_file_path: str,
        prepare_file_path: str | None = None,
    ) -> None:
        module_name = self.module_name
        if not module_name:
            return

        module = __import__(module_name, fromlist=[self.function_name])
        optimizer_function = getattr(module, self.function_name)
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

    try:
        adapter = OptimizerAdapter(
            module_name=context.config.optimizer_module,
            function_name=context.config.optimizer_function,
        )
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
            validate_python_source(after_text)
            snapshot_path = context.paths.artifacts_dir / \
                f"{algorithm_artifact.path.stem}.optimized.py"
            shutil.copy2(algorithm_artifact.path, snapshot_path)
            context.artifacts["algorithm_optimized"] = snapshot_path
            context.log_event("tool_call", "optimizer_finish", {
                              "changed": True, "snapshot_path": snapshot_path})
            return True, "optimizer modified algorithm file in place"

        context.log_event("tool_call", "optimizer_finish", {"changed": False})
        return True, "optimizer completed without changing the algorithm file"
    except Exception as exc:
        algorithm_artifact.path.write_text(before_text, encoding="utf-8")
        context.log_event(
            "tool_call",
            "optimizer_error",
            {"algorithm_path": str(algorithm_artifact.path), "error": str(exc)[:400]},
        )
        return False, f"optimizer failed and restored original algorithm: {exc}"
