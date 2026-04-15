from __future__ import annotations

from .context import PipelineContext
from .models import PipelineRequest, PipelineResult, StageName, StageStatus
from .stages import codegen_stage, evaluate_stage, execution_stage, optimize_stage, package_stage, research_stage


class PipelineRunner:
    def run(self, request: PipelineRequest) -> PipelineResult:
        context = PipelineContext.create(request)
        context.copy_input_image()
        context.log_event("pipeline", "run_start", {"image_path": request.image_path, "scene_prompt": request.scene_prompt})
        result = PipelineResult(run_id=context.run_id, request=request)
        current_stage = StageName.RESEARCH

        try:
            current_stage = StageName.RESEARCH
            context.start_stage(StageName.RESEARCH, "building candidate methods")
            research_result = research_stage(context)
            context.finish_stage(StageName.RESEARCH, StageStatus.SUCCEEDED, "research completed")
            result.research_result = research_result

            current_stage = StageName.CODEGEN
            context.start_stage(StageName.CODEGEN, "generating algorithm file")
            algorithm_artifact = codegen_stage(context, research_result)
            context.finish_stage(StageName.CODEGEN, StageStatus.SUCCEEDED, "code generation completed", artifacts=[algorithm_artifact.path])
            result.algorithm_artifact = algorithm_artifact

            current_stage = StageName.OPTIMIZER
            context.start_stage(StageName.OPTIMIZER, "running in-place optimizer")
            optimizer_success, optimizer_message = optimize_stage(context, algorithm_artifact)
            optimizer_status = StageStatus.SUCCEEDED if optimizer_success else StageStatus.FAILED
            context.finish_stage(StageName.OPTIMIZER, optimizer_status, optimizer_message)
            if not optimizer_success and not request.config.continue_on_optimizer_failure:
                raise RuntimeError(optimizer_message)

            current_stage = StageName.EXECUTOR
            context.start_stage(StageName.EXECUTOR, "running optimized algorithm")
            execution_result = execution_stage(context, algorithm_artifact)
            executor_status = StageStatus.SUCCEEDED if execution_result.success else StageStatus.FAILED
            context.finish_stage(StageName.EXECUTOR, executor_status, "execution finished" if execution_result.success else "execution failed")
            result.execution_result = execution_result
            if not execution_result.success:
                raise RuntimeError("execution stage failed")

            current_stage = StageName.EVALUATOR
            context.start_stage(StageName.EVALUATOR, "computing quality metrics")
            quality_report = evaluate_stage(context, execution_result)
            context.finish_stage(StageName.EVALUATOR, StageStatus.SUCCEEDED, "evaluation completed")
            result.quality_report = quality_report

            manifest_path = context.paths.manifest_path
            result.manifest_path = context.paths.manifest_path
            context.start_stage(StageName.PACKAGE, "writing manifest")
            context.finish_stage(StageName.PACKAGE, StageStatus.SUCCEEDED, "manifest written", artifacts=[manifest_path])
            result.stage_records = list(context.stage_records.values())
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())
            context.log_event("pipeline", "run_success", {"manifest_path": manifest_path})
            return result
        except Exception as exc:
            result.error_code = type(exc).__name__
            result.error_message = str(exc)
            context.log_event("pipeline", "run_error", {"stage": current_stage.value, "error": str(exc), "error_code": type(exc).__name__})
            context.finish_stage(current_stage, StageStatus.FAILED, str(exc), error_code=type(exc).__name__)
            result.stage_records = list(context.stage_records.values())
            manifest_path = context.paths.manifest_path
            result.manifest_path = context.paths.manifest_path
            context.start_stage(StageName.PACKAGE, "writing manifest after failure")
            context.finish_stage(StageName.PACKAGE, StageStatus.SUCCEEDED, "manifest written after failure", artifacts=[manifest_path])
            result.stage_records = list(context.stage_records.values())
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())
            context.log_event("pipeline", "run_failed", {"manifest_path": manifest_path, "error_code": result.error_code})
            return result
