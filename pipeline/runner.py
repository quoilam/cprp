from __future__ import annotations

from pathlib import Path
from typing import Any

from .context import PipelineContext
from .models import (
    DEFAULT_CONTINUE_ON_OPTIMIZER_FAILURE,
    DEFAULT_MAX_BRANCHES,
    DEFAULT_BYPASS_AUTORESEARCH,
    DEFAULT_SELECTION_STRATEGY,
    PipelineConfig,
    PipelineRequest,
    PipelineResult,
    QualityReport,
    StageName,
    StageStatus,
)
from .stages import (
    codegen_stage,
    evaluate_stage,
    execution_stage,
    optimize_stage,
    package_stage,
    research_stage,
    # Multi-branch functions
    codegen_multi_stage,
    optimize_multi_stage,
    execute_multi_stage,
    evaluate_multi_stage,
    select_best_branch,
)


class PipelineRunner:
    """Pipeline runner that supports both single-branch and multi-branch modes.

    Multi-branch mode is activated when max_branches > 1.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def run(self, request: PipelineRequest) -> PipelineResult:
        """Run the pipeline. Uses multi-branch mode if max_branches > 1."""
        if self.config.max_branches > 1:
            return self._run_multi_branch(request)
        return self._run_single_branch(request)

    # ========================================================================
    # Single-branch mode (original behavior, preserved for backward compatibility)
    # ========================================================================
    def _run_single_branch(self, request: PipelineRequest) -> PipelineResult:
        context = PipelineContext.create(request)
        context.copy_input_image()
        context.print_header()

        result = PipelineResult(run_id=context.run_id, request=request, config=self.config)
        current_stage = StageName.RESEARCH

        try:
            # ---------- 1. Research ----------
            current_stage = StageName.RESEARCH
            context.start_stage(StageName.RESEARCH, "正在检索并生成候选方法...")
            research_result = research_stage(
                context, bypass_autoresearch=self.config.bypass_autoresearch)
            candidate_lines = []
            for idx, cand in enumerate(research_result.candidates, 1):
                candidate_lines.append(
                    f"{idx}. {cand.name} (置信度 {cand.confidence:.2f})")
            candidate_details = "\n".join(candidate_lines)
            extra = {
                "推荐策略": research_result.chosen_strategy,
                "候选算法": f"\n{candidate_details}"
            }
            context.finish_stage(
                StageName.RESEARCH, StageStatus.SUCCEEDED, "研究完成", extra_info=extra)
            result.research_result = research_result

            # ---------- 2. Codegen ----------
            current_stage = StageName.CODEGEN
            context.start_stage(StageName.CODEGEN, "正在生成算法文件...")
            algorithm_artifact = codegen_stage(context, research_result)
            context.finish_stage(
                StageName.CODEGEN, StageStatus.SUCCEEDED,
                f"算法文件已生成: {algorithm_artifact.path.name}",
                artifacts=[algorithm_artifact.path],
                extra_info={"策略名称": algorithm_artifact.strategy_name})
            result.algorithm_artifact = algorithm_artifact

            # ---------- 3. Optimizer ----------
            current_stage = StageName.OPTIMIZER
            context.start_stage(StageName.OPTIMIZER, "正在执行优化器...")
            optimizer_success, optimizer_message = optimize_stage(context, algorithm_artifact)
            optimizer_status = StageStatus.SUCCEEDED if optimizer_success else StageStatus.FAILED
            context.finish_stage(
                StageName.OPTIMIZER, optimizer_status, optimizer_message,
                extra_info={"代码变更": "是" if "modified" in optimizer_message else "否"})
            if not optimizer_success and not self.config.continue_on_optimizer_failure:
                raise RuntimeError(optimizer_message)

            # ---------- 4. Executor ----------
            current_stage = StageName.EXECUTOR
            context.start_stage(StageName.EXECUTOR, "正在运行算法...")
            execution_result = execution_stage(context, algorithm_artifact)
            executor_status = StageStatus.SUCCEEDED if execution_result.success else StageStatus.FAILED
            extra_exec = {
                "输出图片": execution_result.output_image_path.name if execution_result.output_image_path else "无",
                "耗时": f"{execution_result.duration_seconds:.1f}秒"
            }
            context.finish_stage(
                StageName.EXECUTOR, executor_status,
                "执行完成" if execution_result.success else "执行失败", extra_info=extra_exec)
            result.execution_result = execution_result
            if not execution_result.success:
                raise RuntimeError("执行阶段失败")

            # ---------- 5. Evaluator ----------
            current_stage = StageName.EVALUATOR
            context.start_stage(StageName.EVALUATOR, "正在计算质量指标...")
            quality_report = evaluate_stage(context, execution_result)
            extra_eval = {
                "PSNR": f"{quality_report.psnr:.2f} dB" if quality_report.psnr else "N/A",
                "SSIM": f"{quality_report.ssim:.3f}" if quality_report.ssim else "N/A",
                "综合评分": f"{quality_report.score:.3f}"
            }
            context.finish_stage(
                StageName.EVALUATOR, StageStatus.SUCCEEDED, "评估完成", extra_info=extra_eval)
            result.quality_report = quality_report

            # ---------- 6. Package ----------
            current_stage = StageName.PACKAGE
            context.start_stage(StageName.PACKAGE, "正在写入产物清单...")
            manifest_path = context.paths.manifest_path
            context.finish_stage(
                StageName.PACKAGE, StageStatus.SUCCEEDED, "清单已写入",
                artifacts=[manifest_path], extra_info={"清单路径": str(manifest_path)})
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            context.print_footer(True, {
                "output_image": str(execution_result.output_image_path),
                "manifest": str(manifest_path)
            })
            return result

        except Exception as exc:
            result.error_code = type(exc).__name__
            result.error_message = str(exc)
            context._write_event_log("pipeline", "run_error", {
                "stage": current_stage.value,
                "error": str(exc),
                "error_code": type(exc).__name__
            })
            context.finish_stage(
                current_stage, StageStatus.FAILED,
                f"错误: {str(exc)}", error_code=type(exc).__name__)
            result.stage_records = list(context.stage_records.values())

            context.start_stage(StageName.PACKAGE, "写入失败清单...")
            manifest_path = context.paths.manifest_path
            context.finish_stage(
                StageName.PACKAGE, StageStatus.SUCCEEDED, "已写入失败清单",
                artifacts=[manifest_path])
            result.manifest_path = manifest_path
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            context.print_footer(False, {"manifest": str(manifest_path)})
            return result

    # ========================================================================
    # Multi-branch mode
    # ========================================================================
    def _run_multi_branch(self, request: PipelineRequest) -> PipelineResult:
        context = PipelineContext.create(request)
        context.copy_input_image()
        context.print_header()

        result = PipelineResult(
            run_id=context.run_id,
            request=request,
            config=self.config,
        )
        current_stage = StageName.RESEARCH

        try:
            max_branches = self.config.max_branches
            bypass_autoresearch = self.config.bypass_autoresearch
            selection_strategy = self.config.selection_strategy

            # ---------- 1. Research ----------
            current_stage = StageName.RESEARCH
            context.start_stage(StageName.RESEARCH, "正在检索并生成候选方法...")
            research_result = research_stage(
                context, bypass_autoresearch=self.config.bypass_autoresearch)

            candidate_lines = []
            for idx, cand in enumerate(research_result.candidates, 1):
                candidate_lines.append(
                    f"{idx}. {cand.name} (置信度 {cand.confidence:.2f})")
            candidate_details = "\n".join(candidate_lines)
            extra = {
                "推荐策略": research_result.chosen_strategy,
                "候选算法": f"\n{candidate_details}",
                "多分支模式": f"将生成最多 {max_branches} 份代码",
            }
            context.finish_stage(
                StageName.RESEARCH, StageStatus.SUCCEEDED, "研究完成", extra_info=extra)
            result.research_result = research_result

            # ---------- 2. Multi-branch Codegen ----------
            current_stage = StageName.CODEGEN
            context.start_stage(StageName.CODEGEN, f"正在为 {max_branches} 个候选方法生成代码...")
            codegen_branches = codegen_multi_stage(context, research_result, max_branches=max_branches)

            if not codegen_branches:
                raise RuntimeError("多分支代码生成失败，没有成功生成任何代码")

            result.codegen_branches = codegen_branches
            # Also preserve the original single-branch artifact for backward compatibility
            if codegen_branches:
                result.algorithm_artifact = codegen_branches[0].artifact

            context.finish_stage(
                StageName.CODEGEN, StageStatus.SUCCEEDED,
                f"已生成 {len(codegen_branches)} 份算法代码",
                artifacts=[str(b.artifact.path) for b in codegen_branches],
                extra_info={
                    "生成数量": len(codegen_branches),
                    "算法名称": [b.candidate.name for b in codegen_branches],
                    "置信度": [f"{b.candidate.confidence:.2f}" for b in codegen_branches],
                })

            # ---------- 3. Multi-branch Optimization ----------
            current_stage = StageName.OPTIMIZER
            bypass_msg = "(跳过耗时的外部优化)" if bypass_autoresearch else ""
            context.start_stage(StageName.OPTIMIZER, f"正在优化 {len(codegen_branches)} 份代码 {bypass_msg}")
            optimized_results = optimize_multi_stage(
                context, codegen_branches, bypass_autoresearch=bypass_autoresearch)
            result.optimized_branches = optimized_results

            context.finish_stage(
                StageName.OPTIMIZER, StageStatus.SUCCEEDED,
                f"已完成 {len(optimized_results)} 份代码优化",
                extra_info={
                    "优化数量": len(optimized_results),
                    "跳过外部优化": str(bypass_autoresearch),
                })

            # ---------- 4. Multi-branch Execution ----------
            current_stage = StageName.EXECUTOR
            context.start_stage(StageName.EXECUTOR,
                                f"正在执行 {len(optimized_results)} 份优化代码...")
            exec_branches = execute_multi_stage(context, optimized_results)
            result.execution_branches = exec_branches

            successful_exec = sum(
                1 for e in exec_branches if e.execution_result.success)
            context.finish_stage(
                StageName.EXECUTOR,
                StageStatus.SUCCEEDED if successful_exec > 0 else StageStatus.FAILED,
                f"执行完成: {successful_exec}/{len(exec_branches)} 成功",
                extra_info={
                    "成功数量": successful_exec,
                    "失败数量": len(exec_branches) - successful_exec,
                })

            if successful_exec == 0:
                raise RuntimeError("所有分支执行均失败")

            # ---------- 5. Multi-branch Evaluation ----------
            current_stage = StageName.EVALUATOR
            context.start_stage(StageName.EVALUATOR, "正在评估各分支质量...")
            quality_branches = evaluate_multi_stage(context, exec_branches)
            result.quality_branches = quality_branches

            # Build comparison table for display
            comparison_lines = []
            for idx, report in quality_branches:
                comparison_lines.append(
                    f"分支{idx}: PSNR={report.psnr:.1f}dB, SSIM={report.ssim:.3f}, 评分={report.score:.3f}")

            context.finish_stage(
                StageName.EVALUATOR, StageStatus.SUCCEEDED,
                f"评估完成: {len(quality_branches)} 份报告",
                extra_info={"对比结果": "\n".join(comparison_lines)})

            # ---------- Select Best Branch ----------
            selection = select_best_branch(
                context, exec_branches, quality_branches,
                selection_strategy=selection_strategy)

            if selection is not None:
                best_idx, best_exec, best_quality = selection
                result.selected_branch_index = best_idx
                result.selected_strategy_name = best_exec.strategy_name
                result.selected_output_image = best_exec.execution_result.output_image_path
                # Also set the legacy fields to point to the best branch
                result.execution_result = best_exec.execution_result
                result.quality_report = best_quality

            # ---------- 6. Package ----------
            current_stage = StageName.PACKAGE
            context.start_stage(StageName.PACKAGE, "正在写入产物清单...")
            manifest_path = context.paths.manifest_path

            # Write detailed branch comparison
            comparison_payload = {
                "branches": [
                    {
                        "candidate_index": idx,
                        "candidate_name": next(
                            (b.candidate.name for b in codegen_branches if b.artifact.candidate_index == idx),
                            "unknown"),
                        "execution": next(
                            (e.to_dict() for e in exec_branches if e.candidate_index == idx),
                            {}),
                        "quality": next(
                            (q.to_dict() for _, q in quality_branches if _ == idx),
                            {}),
                    }
                    for idx in range(max(len(exec_branches), len(quality_branches)))
                ],
                "selected": {
                    "branch_index": result.selected_branch_index,
                    "strategy_name": result.selected_strategy_name,
                    "output_image": str(result.selected_output_image) if result.selected_output_image else None,
                },
            }
            context.write_json("branch_comparison.json", comparison_payload)

            context.finish_stage(
                StageName.PACKAGE, StageStatus.SUCCEEDED, "清单已写入",
                artifacts=[manifest_path], extra_info={"清单路径": str(manifest_path)})
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            # Print summary
            output_images = [
                str(result.selected_output_image)] if result.selected_output_image else []
            context.print_footer(True, {
                "output_image": ", ".join(output_images) if output_images else str(exec_branches[0].execution_result.output_image_path) if exec_branches and exec_branches[0].execution_result.output_image_path else "N/A",
                "selected_branch": f"分支{result.selected_branch_index} ({result.selected_strategy_name})" if result.selected_branch_index is not None else "N/A",
                "manifest": str(manifest_path),
            })
            return result

        except Exception as exc:
            result.error_code = type(exc).__name__
            result.error_message = str(exc)
            context._write_event_log("pipeline", "run_error", {
                "stage": current_stage.value,
                "error": str(exc),
                "error_code": type(exc).__name__,
            })
            context.finish_stage(
                current_stage, StageStatus.FAILED,
                f"错误: {str(exc)}", error_code=type(exc).__name__)
            result.stage_records = list(context.stage_records.values())

            context.start_stage(StageName.PACKAGE, "写入失败清单...")
            manifest_path = context.paths.manifest_path
            context.finish_stage(
                StageName.PACKAGE, StageStatus.SUCCEEDED, "已写入失败清单",
                artifacts=[manifest_path])
            result.manifest_path = manifest_path
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            context.print_footer(False, {"manifest": str(manifest_path)})
            return result
