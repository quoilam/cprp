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
1. 结合外部线索，输出候选方法列表。
2. 每个候选包含方法名、参数建议、理由、来源链接、置信度。
3. 明确指出最终推荐方法。
4. 输出必须是 JSON 对象，字段固定为:
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
		 "summary": "string",
		 "sources": ["url"]
	 }}
5. 不要输出任何 JSON 之外的内容。
""".strip()


CODEGEN_PROMPT_TEMPLATE = """
请根据以下研究结果生成单文件 Python 算法实现。

研究结果:
{research_summary}

要求:
1. 只输出单个 Python 文件内容，不要输出解释文字。
2. 文件必须提供 run(image_path: str, output_path: str, scene_prompt: str) -> dict。
3. 代码必须可执行、可静态检查、无外部网络依赖。
4. 仅允许使用标准库 + Pillow + numpy + scipy + cv2 + skimage + imageio。
5. 保持输出图写入 output_path，返回结果 dict 至少包含 strategy_name、output_path、size。
""".strip()


OPTIMIZER_PROTOCOL = "optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None"

