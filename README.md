# VibeImpl

单入口图像算法自动化流水线的初始实现。

## 运行

```bash
uv run python main.py --image-path path/to/image.png --scene-prompt "请增强夜景照片的清晰度"
```

可选参数包括 `--output-root`、`--session-id`、`--algorithms-root`、`--timeout-seconds`、`--max-memory-mb`、`--optimizer-module` 和 `--continue-on-optimizer-failure`。

默认产物会按 session 聚合写入，并维护最新会话软链接：

```text
output/
	latest -> session_xxxxxxxxxx
	session_xxxxxxxxxx/
		run_.../
			algorithms/
			inputs/
			output/
			stages/
			manifest.json
```
散落到顶层目录；仅当显式传入 `--algorithms-root` 时才写到外部路径。

## 当前阶段
csve
已完成最小闭环骨架：

1. 单入口 CLI。
2. 统一运行上下文与工件目录。
3. 研究、代码生成、优化、执行、评估、打包的稳定阶段契约。
4. 默认使用本地确定性基线算法，后续可替换为 OpenRouter 研究与真实优化器。

