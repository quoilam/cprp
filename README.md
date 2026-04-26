# VibeImpl

单入口图像算法自动化流水线的初始实现。

## 运行

```bash
uv run python main.py --image-path path/to/image.png --scene-prompt "请增强夜景照片的清晰度"
```

现在入口只保留 `--image-path` 和 `--scene-prompt`，其余配置已在代码中固定。

默认产物会按 run 目录写入，并维护 latest 软链接指向最新运行目录：

```text
output/
	latest -> run_xxxxxxxxxx
	run_xxxxxxxxxx/
		original/
		generated/
		result/
		artifacts/
		events.jsonl
		report.json
```

其中：`original/` 存放输入副本，`generated/` 存放算法文件，`result/` 存放处理结果；`report.json` 合并记录各阶段状态与最终结果。

## 当前阶段

已完成最小闭环骨架：

1. 单入口 CLI。
2. 统一运行上下文与工件目录。
3. 研究、代码生成、优化、执行、评估、打包的稳定阶段契约。
4. 默认使用本地确定性基线算法，后续可替换为 OpenRouter 研究与真实优化器。

