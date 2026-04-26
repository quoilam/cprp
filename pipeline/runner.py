from __future__ import annotations

from .context import PipelineContext
from .models import DEFAULT_CONTINUE_ON_OPTIMIZER_FAILURE, PipelineRequest, PipelineResult, StageName, StageStatus
from .stages import codegen_stage, evaluate_stage, execution_stage, optimize_stage, package_stage, research_stage


class PipelineRunner:
    def run(self, request: PipelineRequest) -> PipelineResult:
        context = PipelineContext.create(request)
        context.copy_input_image()

        # 打印流水线头部
        context.print_header()

        result = PipelineResult(run_id=context.run_id, request=request)
        current_stage = StageName.RESEARCH

        try:

            # ---------- 1. 研究阶段 ----------
            current_stage = StageName.RESEARCH
            context.start_stage(StageName.RESEARCH, "正在检索并生成候选方法...")
            research_result = research_stage(context)

            # 构建候选算法详情字符串
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
                StageName.RESEARCH,
                StageStatus.SUCCEEDED,
                "研究完成",
                extra_info=extra
            )
            result.research_result = research_result

            # ---------- 2. 代码生成阶段 ----------
            current_stage = StageName.CODEGEN
            context.start_stage(StageName.CODEGEN, "正在生成算法文件...")
            algorithm_artifact = codegen_stage(context, research_result)
            context.finish_stage(
                StageName.CODEGEN,
                StageStatus.SUCCEEDED,
                f"算法文件已生成: {algorithm_artifact.path.name}",
                artifacts=[algorithm_artifact.path],
                extra_info={"策略名称": algorithm_artifact.strategy_name}
            )
            result.algorithm_artifact = algorithm_artifact

            # ---------- 3. 优化阶段 ----------
            current_stage = StageName.OPTIMIZER
            context.start_stage(StageName.OPTIMIZER, "正在执行优化器...")
            optimizer_success, optimizer_message = optimize_stage(
                context, algorithm_artifact)
            optimizer_status = StageStatus.SUCCEEDED if optimizer_success else StageStatus.FAILED
            context.finish_stage(
                StageName.OPTIMIZER,
                optimizer_status,
                optimizer_message,
                extra_info={
                    "代码变更": "是" if "modified" in optimizer_message else "否"}
            )
            if not optimizer_success and not DEFAULT_CONTINUE_ON_OPTIMIZER_FAILURE:
                raise RuntimeError(optimizer_message)

            # ---------- 4. 执行阶段 ----------
            current_stage = StageName.EXECUTOR
            context.start_stage(StageName.EXECUTOR, "正在运行算法...")
            execution_result = execution_stage(context, algorithm_artifact)
            executor_status = StageStatus.SUCCEEDED if execution_result.success else StageStatus.FAILED
            extra_exec = {
                "输出图片": execution_result.output_image_path.name if execution_result.output_image_path else "无",
                "耗时": f"{execution_result.duration_seconds:.1f}秒"
            }
            context.finish_stage(
                StageName.EXECUTOR,
                executor_status,
                "执行完成" if execution_result.success else "执行失败",
                extra_info=extra_exec
            )
            result.execution_result = execution_result
            if not execution_result.success:
                raise RuntimeError("执行阶段失败")

            # ---------- 5. 评估阶段 ----------
            current_stage = StageName.EVALUATOR
            context.start_stage(StageName.EVALUATOR, "正在计算质量指标...")
            quality_report = evaluate_stage(context, execution_result)
            extra_eval = {
                "PSNR": f"{quality_report.psnr:.2f} dB" if quality_report.psnr else "N/A",
                "SSIM": f"{quality_report.ssim:.3f}" if quality_report.ssim else "N/A",
                "综合评分": f"{quality_report.score:.3f}"
            }
            context.finish_stage(
                StageName.EVALUATOR,
                StageStatus.SUCCEEDED,
                "评估完成",
                extra_info=extra_eval
            )
            result.quality_report = quality_report

            # ---------- 6. 打包阶段 ----------
            current_stage = StageName.PACKAGE
            context.start_stage(StageName.PACKAGE, "正在写入产物清单...")
            manifest_path = context.paths.manifest_path
            context.finish_stage(
                StageName.PACKAGE,
                StageStatus.SUCCEEDED,
                "清单已写入",
                artifacts=[manifest_path],
                extra_info={"清单路径": str(manifest_path)}
            )
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            # 打印成功尾部
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
            # 标记当前阶段失败
            context.finish_stage(
                current_stage,
                StageStatus.FAILED,
                f"错误: {str(exc)}",
                error_code=type(exc).__name__
            )
            result.stage_records = list(context.stage_records.values())

            # 打包阶段（失败后仍写入清单）
            context.start_stage(StageName.PACKAGE, "写入失败清单...")
            manifest_path = context.paths.manifest_path
            context.finish_stage(
                StageName.PACKAGE,
                StageStatus.SUCCEEDED,
                "已写入失败清单",
                artifacts=[manifest_path]
            )
            result.manifest_path = manifest_path
            manifest_path = package_stage(context, result.to_dict())
            result.manifest_path = manifest_path
            result.stage_records = list(context.stage_records.values())

            # 打印失败尾部
            context.print_footer(False, {"manifest": str(manifest_path)})
            return result
