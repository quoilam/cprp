# Development Notes

## 1. 当前代码基准

当前真实实现以 `pipeline/` 包为准：

- [pipeline/runner.py](../pipeline/runner.py#L1)
- [pipeline/context.py](../pipeline/context.py#L1)
- [pipeline/models.py](../pipeline/models.py#L1)
- [pipeline/stages/](../pipeline/stages)

根目录下的 `runner.py`、`context.py`、`stages.py` 不是当前主入口，不应作为新开发的优先修改点。

## 2. 开发原则

### 2.1 只在主链路上加行为

新增业务逻辑优先放在以下位置：

- 流程编排：`pipeline/runner.py`
- 上下文与日志：`pipeline/context.py`
- 数据契约：`pipeline/models.py`
- 阶段逻辑：`pipeline/stages/*.py`
- 重试与韧性：`pipeline/resilience.py`

避免在根目录兼容文件和 `pipeline/` 实现上同时改两份逻辑。

### 2.2 保持阶段顺序稳定

执行顺序必须保持：

1. `research`
2. `codegen`
3. `optimizer`
4. `executor`
5. `evaluator`
6. `package`

阶段内部可以并行处理多个候选，但阶段级顺序不要打乱。

### 2.3 package 不可跳过

无论成功还是失败，都必须执行 `package_stage()` 并落 `report.json`。

### 2.4 失败优先可见

如果一个阶段失败：

- 必须记录 `events.jsonl`
- 必须落到 `report.json`
- 必须尽量保留能帮助排障的中间工件

## 3. 当前核心契约

## 3.1 PipelineRequest

定义：

```python
@dataclass(slots=True)
class PipelineRequest:
    image_path: Path
    scene_prompt: str
```

注意：

- `PipelineRequest` 当前不再携带 config
- config 由 `PipelineRunner` 注入 `PipelineContext`

## 3.2 PipelineConfig

定义位置：

- [pipeline/models.py](../pipeline/models.py#L48)

当前字段：

- `bypass_autoresearch`
- `selection_strategy`
- `optimizer_module`
- `optimizer_function`
- `continue_on_optimizer_failure`
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

说明：

这些配置现在已经通过 `PipelineContext.config` 传递到主要阶段，不再只是静态常量。

## 3.3 生成算法契约

算法文件必须实现：

```python
def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    ...
```

并支持：

```bash
python algo.py --image-path ... --output-path ... --scene-prompt ...
```

当前校验包含：

- Python 语法可解析
- `run()` 签名正确
- 导入白名单正确
- CLI 可执行且能产出输出图

## 3.4 优化器契约

优化器函数签名：

```python
def optimize(
    algo_file_path: str,
    user_prompt: str,
    user_image_file_path: str,
    prepare_file_path: str | None = None,
) -> None:
    ...
```

说明：

- `OptimizerAdapter` 会按 `PipelineConfig.optimizer_module` 和 `optimizer_function` 动态加载
- 优化失败后当前实现会恢复原文件

## 4. 产物约束

运行目录固定结构：

```text
output/<run_id>/
  original/
  generated/
  result/
  artifacts/
  events.jsonl
  report.json
```

常见文件：

- `research.json`
- `research_web_clues.json`
- `codegen.json`
- `optimize.json`
- `execution.json`
- `quality.json`

不要随意改动这些核心文件名，除非同步更新消费方与文档。

## 5. 当前业务逻辑摘要

## 5.1 默认运行偏本地确定性

默认配置下：

- `bypass_autoresearch = True`

因此默认流程更像：

1. 可选做 Web 检索
2. 候选算法优先由本地启发式生成
3. codegen 优先使用本地模板
4. optimize 默认跳过外部 autoresearch
5. 执行候选并评估选优

## 5.2 评估逻辑已任务感知

当前评估不是简单比较输入/输出是否相似，而是根据 `scene_prompt` 推断任务目标：

- 裁剪：构造目标裁剪图
- 放大：构造 resize 后目标图
- 其他：默认以原图为目标

这能避免裁剪类任务在评估时出现偏差。

## 5.3 优化失败处理

- `optimize_stage()` 返回 `(success, message)`，如果失败且 `continue_on_optimizer_failure=False`，runner 直接中止
- 当允许继续时，恢复到先前可执行版本并继续后续阶段

## 6. 韧性与重试

统一实现：

- [pipeline/resilience.py](../pipeline/resilience.py#L1)

适用范围：

- Tavily HTTP 检索
- OpenRouter JSON / code 调用

原则：

- retryable 错误统一由 `build_retry_decorator()` 包装
- 400 系非限流错误快速失败
- 429 / 5xx / 网络错误允许重试
- 终态失败要写日志

不要在各阶段重新手写分散式重试。

## 7. 测试现状

当前测试位于：

- [tests/test_stage_workflow.py](../tests/test_stage_workflow.py#L1)
- [tests/test_stage_execute.py](../tests/test_stage_execute.py#L1)
- [tests/test_stage_select.py](../tests/test_stage_select.py#L1)
- [tests/test_stage_common_validation.py](../tests/test_stage_common_validation.py#L1)

当前已覆盖的重点：

- 研究阶段落盘
- codegen 与优化的旁路与失败恢复
- 执行阶段超时配置读取
- 评估与选优流程
- 导入和签名校验

建议保持测试策略：

1. 阶段函数尽量单测
2. 关键业务偏差要补回归测试
3. 端到端行为以 `PipelineRunner` 和 CLI 冒烟验证为主

## 8. 最近已修正的关键问题

当前代码已修复以下问题：

1. `PipelineConfig` 只定义不生效的问题
2. `bypass_autoresearch=True` 时 codegen 仍强依赖外部 LLM 的问题
3. 优化器失败后没有可靠恢复的问题
4. 裁剪类任务在评估时被错误选优的问题
5. package 阶段先标成功、后写盘的问题

## 9. 后续建议

最值得继续推进的方向：

1. 统一 `research` / `prepare` / `evaluate` 的指标协议
2. 增强 `run()` 返回值与 CLI stdout 的契约校验
3. 清理根目录旧兼容文件，减少双轨实现
4. 给复杂场景增加更多任务感知评估策略
5. 如果未来接真实外部 agent，再补更完整的 optimizer 成功判定
