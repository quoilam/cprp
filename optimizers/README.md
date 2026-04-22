# optimizers

自动优化器模块。利用 Claude CLI 作为自主研究者，在限定时间 budget 内迭代改进传统图像处理算法（去噪、增强等），并保留最优结果。

## 职责

- 对外暴露 `optimize(algo_file_path, user_prompt, user_image_file_path)` 接口，适配 pipeline 的 `OptimizerAdapter`。
- 通过 `program.md` 协议驱动 Claude 自动实验：备份 → 修改代码 → 运行评估 → 保留/回退。
- 测试阶段仅最多 2 轮迭代，单轮 2 分钟超时（可在program.md中修改）

## 结构

```
optimizers/
├── autoresearch.py      # 核心逻辑：填充模板、启动 CLI、实验后还原
├── data/
│   ├── program.md       # Claude 实验协议（prompt 模板）
│   ├── prepare.py       # 评估指标：MSE / PSNR / MAE（后续可扩展）
│   ├── run_exp.ps1      # Windows 执行脚本（可实时跟踪Agent行为）
│   └── run_exp.sh       # Linux / macOS 执行脚本（mac系统脚本没有测试过）
└── __init__.py          # 导出 optimize()
```

