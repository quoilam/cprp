# 项目架构

## 1. 项目定位

VibeImpl 是一个单入口的图像算法自动化流水线。用户输入一张图片和一段场景描述，系统按照固定阶段完成研究、代码生成、优化、执行、评估、选优和打包，并把全过程工件落到 `output/<run_id>/`。

项目核心不是训练模型，而是编排一条可审计、可回退、可重试的图像处理链路：

1. 先把自然语言任务转换成候选算法 clue。
2. 再把候选 clue 转成可执行 Python 算法。
3. 对算法做优化与执行。
4. 基于质量与耗时选择最优结果。
5. 将日志、阶段状态和最终结果统一打包。

## 2. 当前目录与职责

### 2.1 入口层

- [main.py](../main.py#L1)

职责：

- 加载 `.env`
- 解析 CLI 参数
- 生成 `PipelineConfig`
- 构造 `PipelineRequest`
- 调用 `PipelineRunner`
- 输出最终 `PipelineResult`

### 2.2 编排层

- [pipeline/runner.py](../pipeline/runner.py#L1)

职责：

- 决定运行参数与执行策略
- 严格按阶段顺序推进
- 维护 `PipelineResult`
- 记录阶段开始/结束
- 捕获异常并保证 `package_stage` 落盘

### 2.3 运行上下文层

- [pipeline/context.py](../pipeline/context.py#L1)

职责：

- 生成 `run_id` / `session_id`
- 创建 `output/<run_id>/` 目录结构
- 写 `events.jsonl`
- 写阶段检查点 `report.json`
- 提供 `copy_input_image()` / `write_json()` 等上下文能力
- 维护 `stage_records` 与运行期 `artifacts`

### 2.4 数据模型层

- [pipeline/models.py](../pipeline/models.py#L1)

职责：

- 定义系统配置与数据契约
- 承载跨阶段传递的结果对象
- 统一序列化输出结构

关键模型：

- `PipelineConfig`
- `PipelineRequest`
- `ResearchResult`
- `GeneratedAlgorithmArtifact`
- `OptimizedArtifact`
- `ExecutionResult`
- `QualityReport`
- `PipelineResult`
- `StageRecord`

### 2.5 阶段实现层

目录：

- [pipeline/stages/research.py](../pipeline/stages/research.py#L1)
- [pipeline/stages/codegen.py](../pipeline/stages/codegen.py#L1)
- [pipeline/stages/optimize.py](../pipeline/stages/optimize.py#L1)
- [pipeline/stages/execute.py](../pipeline/stages/execute.py#L1)
- [pipeline/stages/evaluate.py](../pipeline/stages/evaluate.py#L1)
- [pipeline/stages/select.py](../pipeline/stages/select.py#L1)
- [pipeline/stages/package.py](../pipeline/stages/package.py#L1)
- [pipeline/stages/common.py](../pipeline/stages/common.py#L1)

说明：

旧的 `pipeline/stages.py` 已被拆分为 `pipeline/stages/` 包。当前真实实现以拆分后的模块为准。

### 2.6 韧性与重试层

- [pipeline/resilience.py](../pipeline/resilience.py#L1)

职责：

- 统一 LLM / HTTP 重试策略
- 区分 retryable / non-retryable 异常
- 记录 retry 事件

### 2.7 外部工具与适配层

- [optimizers/autoresearch.py](../optimizers/autoresearch.py#L1)
- [tools/web_search.py](../tools/web_search.py#L1)

说明：

- `optimizers/autoresearch.py` 是默认外部优化器适配层。
- `tools/web_search.py` 属于工具侧封装，不是当前主流水线的直接入口。

## 3. 运行模式

调用路径：

`PipelineRunner.run()` -> `_run()`

阶段顺序：

1. `research`
2. `codegen`
3. `optimizer`
4. `executor`
5. `evaluator`
6. `package`


## 4. 端到端调用链

完整入口链路：

`main.py`
-> `PipelineConfig.from_args()`
-> `PipelineRequest(...)`
-> `PipelineRunner.run()`
-> `PipelineContext.create()`
-> `PipelineContext.copy_input_image()`
-> `research_stage()`
-> `codegen_stage()`
-> `optimize_stage()`
-> `execution_stage()`
-> `evaluate_stage()`
-> `select_best_result()`
-> `package_stage()`

## 5. 分阶段业务逻辑

## 5.1 Research

文件：

- [pipeline/stages/research.py](../pipeline/stages/research.py#L1)

输入：

- `context.request.scene_prompt`
- 可选 Tavily Web 检索结果

逻辑：

1. 如果存在 `TAVILY_API_KEY`，先检索 Web clue。
2. 如果 `bypass_autoresearch=True`，直接走本地启发式 `_build_local_research_result()`。
3. 否则调用 OpenRouter 输出结构化候选。
4. 写出 `research.json` 和 `research_web_clues.json`。

当前默认行为：

- 默认 `bypass_autoresearch=True`
- 默认会生成若干本地候选

## 5.2 Codegen

文件：

- [pipeline/stages/codegen.py](../pipeline/stages/codegen.py#L1)

输入：

- `ResearchResult`
- Web clue 上下文

逻辑：

1. 根据候选方法生成算法文件路径。
2. 如果 `bypass_autoresearch=True`，直接走本地确定性模板生成。
3. 否则调用 OpenRouter 生成算法源码。
4. 校验：
   - Python 语法
   - `run(image_path, output_path, scene_prompt)` 签名
   - allowed imports
   - CLI 契约可执行
5. 生成可选 `prepare.py`
6. 写入 `codegen.json`，用于记录生成的算法信息。

## 5.3 Optimize

文件：

- [pipeline/stages/optimize.py](../pipeline/stages/optimize.py#L1)

输入：

- 生成的算法文件
- 用户 prompt
- 输入图路径

逻辑：

1. 通过 `OptimizerAdapter` 动态加载优化器。
2. 调用优化协议 `optimize(...)`。
3. 对优化前后源码做 hash 比较。
4. 如果修改过，保存 `.optimized.py` 快照。
5. 如果优化器抛异常：
   - 恢复优化前源码
   - 回退到先前可执行版本
   - 根据 `continue_on_optimizer_failure` 决定继续还是中止

说明：

- `OptimizedArtifact.optimizer_success` 用于显式标记优化器是否成功。

## 5.4 Execute

文件：

- [pipeline/stages/execute.py](../pipeline/stages/execute.py#L1)

逻辑：

1. 用 `sys.executable` 子进程运行算法脚本。
2. 将输入图、副产物输出路径、scene prompt 作为 CLI 参数传入。
3. 执行超时取自 `context.config.executor_timeout_seconds`。
4. 可选内存限制取自 `context.config.max_memory_mb`。
5. 产出 `ExecutionResult`。

工件：

- `execution.json`

## 5.5 Evaluate

文件：

- [pipeline/stages/evaluate.py](../pipeline/stages/evaluate.py#L1)

逻辑：

1. 打开输入图与输出图。
2. 根据 `scene_prompt` 推导任务目标图：
   - 裁剪任务：构造中心裁剪目标
   - 放大任务：构造 resize 后目标
   - 其他任务：默认以原图为目标
3. 计算 `PSNR`、`SSIM`、`latency`
4. 生成综合 `score`
5. 在多候选场景下按候选索引和 `QualityReport` 汇总结果

说明：

评估逻辑已经从“输出越像输入越好”升级为“尽量贴合推断出的任务目标”，以避免裁剪类任务在选优时被误伤。

## 5.6 Select Best Result

文件：

- [pipeline/stages/select.py](../pipeline/stages/select.py#L1)

支持策略：

- `best_score`
- `highest_confidence`
- `first_success`

逻辑：

- `best_score`：选最高评分
- `highest_confidence`：优先置信度最高且执行成功的候选
- `first_success`：按候选顺序选第一个执行成功的结果

## 5.7 Package

文件：

- [pipeline/stages/package.py](../pipeline/stages/package.py#L1)

逻辑：

1. 汇总 `PipelineResult.to_dict()`
2. 写入 `report.json`
3. 保证失败路径也会执行打包

当前实现中，`package_stage()` 的调用时序已调整为“先真实写盘，再标记 package 成功”。

## 6. 输出工件布局

运行目录结构：

```text
output/
  latest -> run_xxx
  run_xxx/
    original/
    generated/
    result/
    artifacts/
    events.jsonl
    report.json
    research.json
    research_web_clues.json
    codegen.json
    optimize.json
    execution.json
    quality.json
```

说明：

- `report.json` 是最终汇总入口。
- `events.jsonl` 是逐事件日志。
- `generated/` 下保留生成脚本与 `.optimized.py` 快照。
- `result/` 下保留算法输出图片。

## 7. 关键契约

### 7.1 生成算法契约

必须提供：

```python
def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    ...
```

同时必须支持 CLI：

```bash
python algo.py --image-path ... --output-path ... --scene-prompt ...
```

### 7.2 优化器契约

当前优化协议：

```python
def optimize(
    algo_file_path: str,
    user_prompt: str,
    user_image_file_path: str,
    prepare_file_path: str | None = None,
) -> None:
    ...
```

### 7.3 阶段状态契约

阶段名固定为：

- `research`
- `codegen`
- `optimizer`
- `executor`
- `evaluator`
- `package`

阶段状态固定为：

- `pending`
- `running`
- `succeeded`
- `failed`
- `skipped`

## 8. 配置来源

配置定义：

- [pipeline/models.py](../pipeline/models.py#L48)

CLI 参数来源：

- [main.py](../main.py#L13)

当前可配置项包括：

- `bypass_autoresearch`
- `selection_strategy`
- `continue_on_optimizer_failure`
- `optimizer_module`
- `optimizer_function`
- `llm_max_retries`
- `http_max_retries`
- `retry_initial_delay`
- `retry_max_delay`
- `retry_jitter`
- `llm_timeout_seconds`
- `http_timeout_seconds`
- `executor_timeout_seconds`
- `executor_retry_once`
- `max_memory_mb`

其中部分参数目前未暴露到 CLI，但已经在 `PipelineConfig` 和 `PipelineContext` 中打通。

## 9. 当前实现特征

### 9.1 默认行为偏稳态

- 默认 `bypass_autoresearch = True`
- 默认 `selection_strategy = best_score`

这意味着当前默认行为是：

1. 研究阶段可检索 Web，但候选方案默认本地生成
2. 代码生成默认走本地确定性模板
3. 优化阶段默认跳过外部 autoresearch
4. 按候选执行、评估、选优和打包

### 9.2 旧兼容层仍存在

根目录下仍然保留：

- `runner.py`
- `context.py`
- `stages.py`

这些文件更接近旧实现或兼容层。当前主入口链路以 `pipeline/` 包为准。

## 10. 建议阅读顺序

如果要快速理解当前项目，建议按这个顺序读：

1. [README.md](../README.md)
2. [main.py](../main.py#L1)
3. [pipeline/models.py](../pipeline/models.py#L1)
4. [pipeline/context.py](../pipeline/context.py#L1)
5. [pipeline/runner.py](../pipeline/runner.py#L1)
6. `pipeline/stages/` 下各阶段实现
7. [pipeline/resilience.py](../pipeline/resilience.py#L1)
8. [optimizers/autoresearch.py](../optimizers/autoresearch.py#L1)
