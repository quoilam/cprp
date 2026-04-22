# 项目架构分析

## 1. 项目定位

VibeImpl 是一个单入口的图像算法自动化流水线。用户只提供一张输入图片和一段场景描述，系统就会按固定顺序完成研究、代码生成、优化、执行、评估和结果打包，最终输出一个带完整过程记录的 run 目录。

这个项目的核心特点不是“训练模型”，而是“编排一条可审计的图像处理生成链路”：前半段负责把自然语言任务转成候选方案和可执行代码，后半段负责把代码跑起来、计算指标、再把结果固化到磁盘。

## 2. 系统分层

### 2.1 入口层

入口在 [main.py](../main.py#L10)。它负责加载环境变量、解析 CLI 参数、封装成 `PipelineRequest` 和 `PipelineConfig`，最后交给 `PipelineRunner` 执行。这个层本身不做业务判断，只是把命令行输入转换成统一请求对象。

### 2.2 编排层

编排核心在 [pipeline/runner.py](../pipeline/runner.py#L8)。`PipelineRunner.run()` 以串行方式驱动整条流水线，按固定顺序调用 `research -> codegen -> optimizer -> executor -> evaluator -> package`。它还负责维护 `PipelineResult`、阶段状态、异常捕获和最终返回码语义。

### 2.3 状态与目录层

[pipeline/context.py](../pipeline/context.py#L45) 负责 run 级别的目录结构、事件日志和阶段记录。`PipelineContext` 会创建 `session_dir`、`run_dir`、`algorithms/`、`inputs/`、`artifacts/`、`stages/`、`logs/`、`output/` 等目录，并把每个阶段的状态落盘为 JSON。

### 2.4 数据模型层

[pipeline/models.py](../pipeline/models.py#L9) 定义了系统中的主要数据契约：`PipelineRequest`、`PipelineConfig`、`ResearchResult`、`GeneratedAlgorithmArtifact`、`ExecutionResult`、`QualityReport`、`StageRecord` 和 `PipelineResult`。这些结构是整个项目在“研究结果 -> 代码生成 -> 执行结果 -> 评估报告 -> manifest”链路中的数据骨架。

### 2.5 阶段实现层

真正的业务逻辑集中在 [pipeline/stages.py](../pipeline/stages.py#L260)。其中：

- `research_stage()` 调 OpenRouter 和可选 Tavily，生成结构化候选方法。
- `codegen_stage()` 调 OpenRouter 生成单文件 Python 算法，并做语法、签名、导入约束和子进程契约验证。
- `optimize_stage()` 通过配置的优化器模块对生成代码做原地改写。
- `execution_stage()` 用子进程运行算法文件，产出最终图片。
- `evaluate_stage()` 比较输入图和输出图，计算 PSNR、SSIM、延迟和综合评分。
- `package_stage()` 将结果写入 `manifest.json`。

### 2.6 外部优化器层

`optimizers/autoresearch.py` 是一个“外部实验适配器”。它不直接属于主流水线核心，但会被 `OptimizerAdapter` 动态导入并执行。它的作用是把主流水线生成的算法文件交给外部脚本/实验协议继续迭代。

### 2.7 评估实验层

[optimizers/data/prepare.py](../optimizers/data/prepare.py#L1) 提供了另一个独立的评估脚本，定义了 MSE、PSNR、MAE 等指标。它更像实验协议的一部分，而不是主流水线真正使用的评估器。

## 3. 端到端业务流程

### 3.1 输入阶段

用户执行 [main.py](../main.py#L10) 后，必须提供 `--image-path` 和 `--scene-prompt`。其余参数主要影响输出目录、超时、内存限制和优化器插件选择。

### 3.2 研究阶段

`PipelineRunner` 先调用 [pipeline/runner.py](../pipeline/runner.py#L17) 中的 `research_stage()`。这个阶段会：

- 读取环境变量中的 `OPENROUTER_API_KEY` 和 `OPENROUTER_MODEL`。
- 如果存在 `TAVILY_API_KEY`，先抓取 web 线索。
- 将场景描述和 web 线索拼装成研究提示词。
- 让模型输出 JSON 结构的候选方法。
- 把研究结果写入 `research.json` 和 `research_web_clues.json`。

### 3.3 代码生成阶段

研究结果进入 `codegen_stage()` 后，会生成一个单文件 Python 算法。该阶段除了调用模型，还会做几层约束：

- 语法解析，确保源码可被 `ast.parse()` 接受。
- 入口签名检查，要求 `run(image_path, output_path, scene_prompt)`。
- 导入白名单检查，只允许标准库和少量图像处理库。
- 子进程契约验证，要求脚本能在命令行被运行并写出输出图。

通过后，算法源文件会写入 `algorithms/` 或外部指定目录，并生成 `codegen.json`。

### 3.4 优化阶段

优化阶段由 `OptimizerAdapter` 接管。若配置了 `--optimizer-module`，系统会动态导入该模块并调用 `optimize(algo_file_path, user_prompt, user_image_file_path)`。默认配置指向 `optimizers.autoresearch`，它会基于 `program.md` 协议启动外部实验脚本，对算法文件做进一步尝试。

### 3.5 执行阶段

`execution_stage()` 会用 `sys.executable` 子进程运行生成的算法文件，把 `--image-path`、`--output-path`、`--scene-prompt` 传给算法脚本。它同时支持超时和 POSIX 下的内存限制，并把 stdout、stderr、返回码和输出图路径写入 `execution.json`。

### 3.6 评估阶段

`evaluate_stage()` 会把输入图和输出图转成 RGB 数组，计算 PSNR、SSIM 和延迟，再按固定公式折算成综合分数，最后写入 `quality.json`。

### 3.7 打包阶段

最后，`package_stage()` 把 `PipelineResult.to_dict()` 写入 `manifest.json`，并把所有阶段记录、错误信息、执行结果和质量结果固化下来。这个文件是外部排障和审计的主入口。

## 4. 调用链总览

一次完整运行的控制链路如下：

`main.py` -> `PipelineRunner.run()` -> `PipelineContext.create()` -> `copy_input_image()` -> `research_stage()` -> `codegen_stage()` -> `optimize_stage()` -> `execution_stage()` -> `evaluate_stage()` -> `package_stage()`

其中，`PipelineContext` 负责目录、事件日志和阶段记录，`PipelineRunner` 负责顺序和异常语义，`stages.py` 负责所有业务动作，`models.py` 负责结果结构。

## 5. 关键产物

一次成功运行会生成这些关键产物：

- `inputs/`：输入图片副本。
- `algorithms/`：生成的算法源码。
- `artifacts/`：优化快照、代码生成验证文件等中间物。
- `stages/*.json`：每个阶段的状态记录。
- `logs/events.jsonl`：事件日志。
- `execution.json`：执行结果。
- `quality.json`：评估结果。
- `manifest.json`：最终汇总结果。

## 6. 目前可确认的缺陷和 bug

### 6.1 打包阶段的成功状态早于真实写盘

在 [pipeline/runner.py](../pipeline/runner.py#L52) 里，`package` 阶段先被标记为 `succeeded`，随后才调用 `package_stage()` 写 `manifest.json`。如果真正写盘时出错，异常会落到 `except` 分支，但 `current_stage` 仍然停留在 `evaluator`，导致包装失败被错误归因到评估阶段，而不是 package 阶段。这个问题会让失败日志和阶段状态都失真。

### 6.2 外部优化器失败不会被显式感知

在 [optimizers/autoresearch.py](../optimizers/autoresearch.py#L46) 中，`subprocess.run()` 的返回码没有被检查，`start_experiment()` 最终总是返回 `train_file_path`。这意味着外部实验脚本即使失败、超时或中途报错，只要没有抛异常到 Python 层，主流水线也很难感知优化器其实没成功。

### 6.3 代码生成契约验证过于宽松

[pipeline/stages.py](../pipeline/stages.py#L221) 的 `_verify_generated_algorithm_contract()` 只检查“子进程是否退出 0”和“是否产生输出文件”，但没有验证 `run()` 的返回值结构，也没有检查命令行输出是否符合预期。再加上 [prompts.py](../prompts.py#L44) 里写明 `run()` 应该返回字典，这里实际上存在契约与验证不一致的问题。结果是一些“能跑但不符合约定”的算法也可能被误判为合格。

### 6.4 评估目标与优化协议存在指标漂移

[pipeline/stages.py](../pipeline/stages.py#L526) 的主流水线评估使用 PSNR、SSIM 和延迟综合分，而 [optimizers/data/prepare.py](../optimizers/data/prepare.py#L1) 只定义了 MSE、PSNR、MAE。也就是说，外部优化实验如果依赖 `prepare.py`，它优化的目标和主流水线最终考核的目标并不一致，会造成“实验看起来变好，但主流程评分不升”的现象。

### 6.5 算法生成后立即执行，缺少更强的失败隔离

[pipeline/stages.py](../pipeline/stages.py#L315) 的 codegen 只做了语法、导入和最小契约验证，然后就进入执行阶段。这里没有对 `run()` 返回值结构、图像模式、尺寸、异常类型做更严格的约束，因此“生成代码可执行”并不等于“生成代码对业务目标有效”。这不是单点错误，但会显著增加后续执行与评估阶段的噪声。

## 7. 总结

从架构上看，这个项目已经形成了比较清晰的分层：入口层、编排层、状态层、阶段层、外部优化层和评估层都已具备。真正的核心控制点在 [pipeline/runner.py](../pipeline/runner.py#L8) 和 [pipeline/stages.py](../pipeline/stages.py#L260)，而输出的审计能力主要依赖 [pipeline/context.py](../pipeline/context.py#L45) 的目录与日志约定。

当前最值得优先修的不是“再加一个功能”，而是先把两个闭环补牢：一是 package 阶段的错误归因和落盘顺序，二是优化器与代码生成契约的失败可见性。把这两点修好后，整条流水线的可观测性和可调试性会明显提升。