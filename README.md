# VibeImpl

单入口图像算法自动化流水线。

它的目标是：给定一张输入图和一段场景描述，自动完成研究、代码生成、优化、执行、评估和打包，并把全过程工件落到本地。

## 项目特点

- 单入口 CLI
- 单一执行模式
- 全流程工件可追踪
- 内置 fallback 与 retry
- 支持外部优化器接入

当前主实现位于 `pipeline/` 包下，根目录旧的 `runner.py`、`context.py`、`stages.py` 不再是主入口。

## 快速开始

```bash
uv run python main.py --image-path /abs/path/input.png --scene-prompt "请增强夜景照片的清晰度"
```

## CLI 参数

当前入口参数定义见 [main.py](./main.py#L1)。

主要参数：

- `--image-path`：输入图片路径
- `--scene-prompt`：场景描述
- `--bypass-autoresearch`：旁路外部 autoresearch
- `--no-bypass-autoresearch`：显式启用外部 autoresearch
- `--continue-on-optimizer-failure`：优化失败时是否继续

## 当前默认行为

默认配置来自 [pipeline/models.py](./pipeline/models.py#L1)：


也就是说，当前默认运行偏稳态：

 1. 默认单一执行模式
2. 研究阶段可检索 Web，但候选方案优先本地生成
3. codegen 默认直接走本地确定性模板
4. optimize 默认跳过外部 autoresearch
5. 最后仍会执行、评估并打包

## 阶段顺序

一次完整运行的阶段顺序固定为：

1. `research`
2. `codegen`
3. `optimizer`
4. `executor`
5. `evaluator`
6. `package`

## 当前架构

建议从这几个文件理解项目：

- [main.py](./main.py#L1)：CLI 入口
- [pipeline/runner.py](./pipeline/runner.py#L1)：主编排
- [pipeline/context.py](./pipeline/context.py#L1)：上下文与日志
- [pipeline/models.py](./pipeline/models.py#L1)：数据模型
- [pipeline/stages/](./pipeline/stages)：各阶段实现
- [pipeline/resilience.py](./pipeline/resilience.py#L1)：重试与异常分类

更完整的架构说明见：

- [doc/arch.md](./doc/arch.md)
- [doc/development.md](./doc/development.md)

## 输出工件

默认输出目录结构：

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
```

常见附加工件：

- `research.json`
- `research_web_clues.json`
- `codegen.json`
- `execution.json`
- `quality.json`

## 核心契约

### 生成算法文件

生成算法必须提供：

```python
def run(image_path: str, output_path: str, scene_prompt: str) -> dict:
    ...
```

并支持 CLI：

```bash
python algo.py --image-path ... --output-path ... --scene-prompt ...
```

### 优化器

优化器协议：

```python
def optimize(
    algo_file_path: str,
    user_prompt: str,
    user_image_file_path: str,
    prepare_file_path: str | None = None,
) -> None:
    ...
```

## 评估逻辑说明

当前评估已经是“任务感知”的，而不只是比较输入图和输出图是否相似：

- 裁剪任务：构造中心裁剪目标图
- 放大任务：构造 resize 后目标图
- 其他任务：默认以原图为参考目标

这样做是为了避免裁剪类任务被“最像原图”但不符合目标的结果误导评分。

## 测试

运行测试：

```bash
uv run python -m unittest discover -s tests -q
```

当前测试主要覆盖：

- 阶段落盘
- 优化失败恢复
- 执行配置透传
- 评估逻辑
- 代码生成契约校验

## 开发说明

开发时请优先遵守：

1. 新逻辑优先改 `pipeline/`，不要优先改根目录旧兼容文件
2. `package_stage` 不可跳过
3. 所有失败路径都应可在 `events.jsonl` 和 `report.json` 中定位
4. 尽量保持工件名稳定，避免影响下游消费
5. LLM / HTTP 重试统一复用 `pipeline/resilience.py`

更多开发细节见 [doc/development.md](./doc/development.md)。
