## Plan: 单入口图像算法自动化系统

将项目收敛为单一入口的自动化流水线应用，不再保留聊天模式。系统接收用户图片与场景提示词，自动研究算法、生成本地 Python 算法文件、调用黑箱优化函数进行原地优化（inplace）、执行优化后算法并输出结果与评估信息。优先保证阶段契约稳定、可追溯与可测试。

**Steps**
1. Phase A - 单入口与主流程骨架。
2. 将 [main.py](main.py) 定义为唯一入口，仅支持 pipeline 运行参数（image_path、scene_prompt、可选配置）。
3. 在入口层定义统一运行上下文：run_id、阶段状态、工件目录、错误码，作为后续所有模块共享上下文。
4. Phase B - 阶段契约与数据模型（阻塞后续开发）。
5. 定义关键模型：PipelineRequest、ResearchResult、GeneratedAlgorithmArtifact、ExecutionResult、QualityReport、PipelineResult。
6. 固定阶段 I/O：Research 输出候选方法；Codegen 输出算法文件路径；Optimizer 阶段接收算法文件路径并原地修改；Executor 输出结果图与执行元数据；Evaluator 输出质量与速度指标。
7. 确立算法文件执行契约（统一函数签名）与工件命名规范，确保阶段解耦与可替换。
8. Phase C - Research 与 Codegen 子系统。
9. Research：基于 OpenRouter + Web/GitHub 线索生成结构化候选列表（方法、参数建议、理由、来源链接、置信度）。
10. Codegen：按固定契约生成单文件 Python 算法实现，执行语法与静态安全校验后落盘。
11. 引入提示词模板与响应解析规范，保证模型输出可解析、可重试、可审计。
12. Phase D - 黑箱优化（inplace 函数协议）。
13. 实现 OptimizerAdapter，协议固定为函数调用：optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None。
14. 将优化阶段定义为“原地修改算法文件”：执行前后进行文件快照与校验（存在性、可读性、内容变化可选、语法可解析）。
15. 若优化阶段失败或超时，按策略处理：可配置为终止流程，或回退到优化前快照继续执行。
16. Phase E - 安全执行、评估、结果汇总。
17. Executor 使用子进程运行优化后算法，控制超时、内存与工作目录隔离。
18. Evaluator 计算平衡指标：速度（时延）+ 质量（PSNR/SSIM），输出分项与综合评分。
19. ResultPackager 汇总 run_id 工件：输入元信息、研究摘要、算法文件、结果图、指标 JSON、阶段清单、错误上下文。
20. Phase F - 验证与交付。
21. 单元测试覆盖：研究解析、代码校验、优化器适配器（inplace 调用）、执行器资源限制、评估计算。
22. 集成测试覆盖：单图 happy path（研究 -> 生成 -> 优化 -> 执行 -> 评估）与关键失败路径。
23. 建立验收标准：可重复执行、产物可追溯、阶段失败可定位。

**Relevant files**
- /Users/quoilam/Documents/cprp/vibeimpl/main.py — 唯一入口与总编排启动点。
- /Users/quoilam/Documents/cprp/vibeimpl/prompts.py — 研究与代码生成的提示词模板与输出格式约束。
- /Users/quoilam/Documents/cprp/vibeimpl/tools/web_search.py — 研究阶段的外部信息输入来源。
- /Users/quoilam/Documents/cprp/vibeimpl/doc/development.md — 架构设计、阶段契约、黑箱函数协议与验收标准文档。
- /Users/quoilam/Documents/cprp/vibeimpl/algos — 生成与优化中的算法文件工件目录。
- /Users/quoilam/Documents/cprp/vibeimpl/output — 结果图、日志、指标与 run 级别清单目录。

**Verification**
1. 单入口运行可在给定图片与提示词下完整产出结果图与指标文件。
2. 优化阶段正确调用 inplace 黑箱函数，并在优化后对算法文件完成可执行性校验。
3. 执行阶段可在超时与异常场景下返回明确错误并写入阶段状态。
4. 指标阶段稳定输出速度与质量报告，且结果与 run_id 工件一致。
5. 全流程日志可定位任一失败阶段与上下文。

**Decisions**
- 架构入口：不保留聊天能力，仅保留单一 pipeline 入口。
- 黑箱协议：Python 函数调用，inplace 修改算法文件，返回 None。
- LLM 提供商：OpenRouter。
- 研究范围：外网技术资料与 GitHub 线索。
- 执行隔离：先采用子进程 + 超时 + 内存限制。
- 优化目标：质量与速度平衡。
- 第一版范围：单图、同步、可追溯流水线，不含批处理与分布式。

**Further Considerations**
1. 建议明确 inplace 后的“成功判定规则”：仅以无异常返回为成功，或增加语法通过/功能通过双判定。
2. 建议确定优化阶段超时时间与回退策略默认值，避免流水线行为不确定。
3. 建议在第一版即沉淀统一 manifest 结构，方便后续自动评测与实验对比.
