from __future__ import annotations


PIPELINE_SYSTEM_PROMPT = """
你是一个图像算法自动化流水线。
你接收用户图片与场景提示，输出结构化研究结果、可执行算法文件、执行元数据与评估报告。
所有阶段必须保持可追踪、可重试、可审计。
""".strip()


RESEARCH_PROMPT_TEMPLATE = """
请基于以下场景提示，为图像处理任务生成结构化研究结论。

场景提示:
{scene_prompt}

外部检索线索:
{web_context}

要求:
1. 结合外部线索，同时研究两类事实:
   - 图像处理算法实现思路
   - 效果评价指标与评价方案
2. 输出候选方法列表。
3. 每个候选包含方法名、参数建议、理由、来源链接、置信度。
3. 明确指出最终推荐方法。
4. 输出评价方案，至少包含 metrics 列表与 evaluation_plan。
5. 输出必须是 JSON 对象，字段固定为:
	 {{
		 "candidates": [
			 {{
				 "name": "string",
				 "description": "string",
				 "parameters": {{"key": "value"}},
				 "rationale": "string",
				 "sources": ["url"],
				 "confidence": 0.0
			 }}
		 ],
		 "chosen_strategy": "string",
		 "evaluation_metrics": ["string"],
		 "evaluation_plan": "string",
		 "summary": "string",
		 "sources": ["url"]
	 }}
6. 若外部线索不足，也要给出可执行的默认评价方案。
7. 不要输出任何 JSON 之外的内容。
""".strip()


CODEGEN_PROMPT_TEMPLATE = """
请根据以下研究结果生成单文件 Python 算法实现。

研究结果:
{research_summary}

外部检索线索:
{web_context}

要求:
1. 只输出单个 Python 文件内容，不要输出解释文字。
2. 文件必须提供 run(image_path: str, output_path: str, scene_prompt: str) -> dict。
3. 代码必须可执行、可静态检查、无外部网络依赖。
4. 仅允许使用标准库 + Pillow + numpy + scipy + cv2 + skimage + imageio。
5. 保持输出图写入 output_path，返回结果 dict 至少包含 strategy_name、output_path、size。
""".strip()


PREPARE_CODEGEN_PROMPT_TEMPLATE = """
请根据以下研究结果与外部线索，生成用于图像结果评价的 prepare.py。

研究结果:
{research_summary}

外部检索线索:
{web_context}

要求:
1. 只输出单个 Python 文件内容，不要输出解释文字。
2. 仅允许使用标准库 + Pillow + numpy + scipy + cv2 + skimage + imageio。
3. 文件中必须提供一个函数:
	evaluate(input_image_path: str, output_image_path: str) -> dict
4. evaluate 返回的 dict 必须至少包含:
	- primary_metric_name: str
	- primary_metric_value: float
	- metrics: dict[str, float]
	- higher_is_better: bool
5. 指标和实现要与 research_result.evaluation_metrics / evaluation_plan 一致。
6. 代码应稳健处理不同尺寸图像（必要时对齐尺寸），并避免抛出难以理解的异常。
""".strip()


CODEGEN_PROMPT_TEMPLATE_FOR_CANDIDATE = """
请根据以下研究结果，为特定的候选算法生成单文件 Python 实现。

研究结果:
{research_summary}

目标候选方法 (第{candidate_index}个):
  名称: {candidate_name}
  描述: {candidate_description}
  参数: {candidate_parameters}
  理由: {candidate_rationale}
  置信度: {candidate_confidence}

外部检索线索:
{web_context}

要求:
1. 只输出单个 Python 文件内容，不要输出解释文字。
2. 文件必须提供 run(image_path: str, output_path: str, scene_prompt: str) -> dict。
3. 代码必须可执行、可静态检查、无外部网络依赖。
4. 仅允许使用标准库 + Pillow + numpy + scipy + cv2 + skimage + imageio。
5. 保持输出图写入 output_path，返回结果 dict 至少包含 strategy_name、output_path、size。
6. 请使用候选方法名称作为 STRATEGY_NAME，并实现该候选方法描述的算法逻辑。
""".strip()


OPTIMIZER_PROTOCOL = (
    "optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str, "
    "prepare_file_path: str | None = None) -> None"
)
