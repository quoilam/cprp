# VibeImpl

VibeImpl 是一个单入口的图像算法自动化流水线：输入一张图片和一段场景描述，自动完成研究、代码生成、优化、执行、评估与结果打包。

## 功能概览

- 单命令执行完整流水线：`research -> codegen -> optimizer -> executor -> evaluator -> package`
- 每个阶段都会落盘状态文件（含状态、报错、时间戳、产物）
- 执行阶段通过子进程运行算法，并支持超时与内存限制
- 评估阶段输出 PSNR、SSIM、延迟和综合得分
- 即使中途失败，也会写出最终 `manifest.json` 便于排查

## 环境要求

- Python `>= 3.12`
- 推荐使用 `uv`

安装依赖：

```bash
uv sync
```

## 快速开始

```bash
uv run python main.py \
	--image-path path/to/image.png \
	--scene-prompt "请增强夜景照片的清晰度"
```

运行结束后会在终端输出 JSON 结果，命令返回码规则：

- 成功返回 `0`
- 失败返回 `1`

## CLI 参数

必填参数：

- `--image-path`：输入图片路径
- `--scene-prompt`：任务描述（自然语言）

可选参数：

- `--output-root`：输出根目录，默认 `output`
- `--session-id`：会话 ID，不传则自动生成 `session_<10hex>`
- `--algorithms-root`：算法文件外部目录，不传则写入当前 run 的 `algorithms/`
- `--timeout-seconds`：执行超时秒数，默认 `180`
- `--max-memory-mb`：执行进程内存上限（仅 POSIX 有效）
- `--optimizer-module`：优化器模块名（可选）
- `--optimizer-function`：优化器函数名，默认 `optimize`
- `--continue-on-optimizer-failure`：优化失败后是否继续执行后续阶段

## 输出结构

默认输出目录结构如下：

```text
output/
	latest -> session_xxxxxxxxxx
	session_xxxxxxxxxx/
		run_YYYYMMDDTHHMMSSZ_xxxxxxxxxx/
			algorithms/
			inputs/
			artifacts/
			logs/
				events.jsonl
			output/
			stages/
				research.json
				codegen.json
				optimizer.json (按阶段写入状态，不一定单独命名产物)
				executor.json (按阶段写入状态，不一定单独命名产物)
				evaluator.json (按阶段写入状态，不一定单独命名产物)
				package.json (按阶段写入状态，不一定单独命名产物)
			research.json
			research_web_clues.json
			codegen.json
			execution.json
			quality.json
			manifest.json
```

说明：

- `output/latest` 软链接指向最新 session 目录
- 若 `output/latest` 已是实体目录，程序会保留该目录，不会强制覆盖

## 阶段说明

1. `research`
	 - 结合场景描述（可选叠加 Tavily 线索）生成结构化候选方案
	 - 写出 `research.json`、`research_web_clues.json`

2. `codegen`
	 - 生成单文件 Python 算法实现
	 - 强制校验 `run(image_path, output_path, scene_prompt)` 签名
	 - 子进程执行契约验证，失败最多重试 3 次
	 - 写出 `codegen.json`

3. `optimizer`
	 - 通过 `--optimizer-module` 动态加载优化器
	 - 优化器协议固定为：
		 `optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None`
	 - 若优化后文件变更，会保存 `artifacts/*.optimized.py` 快照

4. `executor`
	 - 以子进程运行生成算法，传入 `--image-path --output-path --scene-prompt`
	 - 超时后标记 `timed_out=true`
	 - 写出 `execution.json`

5. `evaluator`
	 - 基于输入图与输出图计算指标并写出 `quality.json`
	 - 综合评分公式：
		 - `normalized_psnr = min(psnr / 40, 1)`
		 - `latency_score = max(0, 1 - duration_seconds / timeout_seconds)`
		 - `score = mean([normalized_psnr, max(0, ssim), latency_score])`

6. `package`
	 - 将最终结果写入 `manifest.json`
	 - 成功/失败路径都会执行

## 环境变量

研究与代码生成依赖 OpenRouter：

- `OPENROUTER_API_KEY`（必填）
- `OPENROUTER_MODEL`（必填）
- `OPENROUTER_BASE_URL`（可选，默认官方地址）

可选 Web 检索：

- `TAVILY_API_KEY`

如果缺少 OpenRouter 相关变量，`research/codegen` 会失败，但流程仍会写出完整失败清单和 `manifest.json`。

## 相关文档

- 开发说明见 `doc/development.md`

