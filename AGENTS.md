# AGENTS.md

本文件定义本仓库中 AI 编码代理的默认行为。目标是让代理在图像算法自动化流水线中稳定执行以下工作流：

1. 联网检索用户场景相关算法思路与评价指标线索（clues）。
2. 基于两个不同 clue 生成两份候选算法代码。
3. 将两份代码分别交给 autoresearch 进行优化。
4. 使用优化后的代码完成用户指定场景的图像生成。
5. 全流程可观测（日志与工件完整），并具备 fallback 与 retry。

## 1) 项目定位与入口

- 单入口：`main.py`。
- 默认运行命令：

```bash
uv run python main.py --image-path /abs/path/input.png --scene-prompt "你的场景描述"
```

- 多分支推荐命令（用于双 clue 双代码路径）：

```bash
uv run python main.py \
  --image-path /abs/path/input.png \
  --scene-prompt "你的场景描述" \
  --max-branches 2 \
  --no-bypass-autoresearch \
  --selection-strategy best_score
```

## 2) 必须遵守的执行顺序

按以下阶段顺序执行并保留阶段产物：

1. `research_stage`：检索 clue，产出候选算法与评价指标。
2. `codegen_multi_stage`：按候选生成多份代码。
3. `optimize_multi_stage`：逐分支调用优化器，原地修改算法文件。
4. `execute_multi_stage`：逐分支执行算法脚本。
5. `evaluate_multi_stage`：逐分支评估质量与延迟。
6. `select_best_branch`：根据 `selection_strategy` 选优。
7. `package_stage`：写入 `report.json` 与阶段记录。

禁止跳过 `package_stage` 与日志落盘。

## 3) 双 clue / 双代码分支约束

- 当任务要求“两个 clue + 两份代码”时，优先设置 `--max-branches 2`。
- `research_stage` 输出的前两名候选应对应两条分支代码。
- 每个分支必须独立记录：候选信息、代码路径、优化结果、执行结果、质量结果。

## 4) Retry 与 Fallback 策略

- LLM/HTTP 失败：使用 `pipeline/resilience.py` 的重试策略，不要手写分散重试逻辑。
- Web 检索失败（如 Tavily 不可用）：降级为空 clue 继续流水线，并在日志中标注降级原因。
- 优化阶段失败：遵循 `--continue-on-optimizer-failure`。
  - 开启时：回退到未优化或当前可执行版本继续。
  - 关闭时：立即失败并打包失败清单。
- 分支级失败不应直接抛弃整个 run，除非全部分支失败。

## 5) 可观测性与工件要求

每次运行必须写入 `output/<run>/` 下完整工件：

- `events.jsonl`
- `report.json`（合并阶段记录与最终结果）
- `research.json`、`codegen.json`、`execution.json`、`quality.json`
- `generated/*.py`（及可选 `*.prepare.py`）
- `result/*`（生成图像）

事件日志至少覆盖：阶段开始/结束、重试、降级、错误码、最终选优结果。

## 6) 关键契约（不得破坏）

- 生成算法脚本需满足命令行执行契约（由 pipeline 校验）。
- 生成算法核心函数签名：

```python
def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    ...
```

- 优化器协议：

```python
def optimize(
    algo_file_path: str,
    user_prompt: str,
    user_image_file_path: str,
    prepare_file_path: str | None = None,
) -> None:
    ...
```

## 7) 代码边界与首选修改点

- 编排入口：`main.py`
- 流程控制：`pipeline/runner.py`
- 阶段实现：`pipeline/stages.py`
- 重试与韧性：`pipeline/resilience.py`
- 运行上下文与日志：`pipeline/context.py`
- 提示词模板：`prompts.py`
- Web 检索：`tools/web_search.py`
- 优化适配：`optimizers/autoresearch.py`

优先在上述文件内做最小修改，避免跨层重复实现。

## 8) 先链接再复制

详细背景与设计说明请直接参考，不在本文件复制大段内容：

- 项目总览：`README.md`
- 架构说明：`doc/arch.md`
- 开发约束：`doc/development.md`
- 优化器说明：`optimizers/README.md`
- 现有规划提示：`.github/prompts/plan-imageAlgorithmAutomation.prompt.md`

## 9) 代理执行检查清单

在提交改动前，代理应确认：

1. 能跑通一次端到端命令并产出 `report.json`。
2. 分支模式下至少有一条成功执行分支，且有明确选优结果。
3. 失败路径可在 `events.jsonl` 与 `report.json` 中定位。
4. 未引入破坏契约的签名变更。
5. 所有新增行为可在现有目录结构中追溯。
