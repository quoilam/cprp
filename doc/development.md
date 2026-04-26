# Development Notes

## Architecture

The project is being collapsed into a single pipeline entrypoint. The CLI accepts only image-related inputs and produces a run directory with all artifacts, stage manifests, and metrics.

## Current Contracts

- `PipelineRequest` carries the image path, scene prompt, and runtime config.
- `PipelineContext` owns the run ID, artifact paths, and stage records.
- Generated algorithm files must expose `run(image_path: str, output_path: str, scene_prompt: str) -> dict`.
- The optimizer protocol is fixed as `optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None`.

## Artifact Layout

- `output/<run_id>/original/` stores copied input assets.
- `output/<run_id>/generated/` stores generated algorithm files.
- `output/<run_id>/result/` stores execution outputs.
- `output/<run_id>/artifacts/` stores snapshots and stage byproducts.
- `output/<run_id>/events.jsonl` stores event logs.
- `output/<run_id>/report.json` stores merged stage records and the final packaged result.
- `output/latest` is a soft link pointing to the latest `<run_id>` directory.

## Verification Targets

1. One command should run research, code generation, optimizer, executor, evaluator, and packaging.
2. Every stage should record status and errors.
3. The executor must run the algorithm file in a subprocess and enforce a timeout.
4. The evaluator should emit PSNR, SSIM, latency, and a combined score.

## Recent Changes (2026-04-27)

- Commit: 7d72a48
- Summary: Fixed pipeline runtime issues, hardened LLM/OpenRouter handling, added deterministic fallback for codegen, and improved artifact layout and reporting.
- Key fixes and improvements:
	- Normalize OpenRouter base URL and tolerate varied response shapes to avoid parsing errors when the API returns different payload formats.
	- Detect and reject HTML responses from LLM endpoints early with clear error messages.
	- Add contract enforcement for generated algorithms: require `run(image_path, output_path, scene_prompt)` and provide a deterministic local fallback when LLM outputs fail verification.
	- Fix executor output filename bug (avoid using `.py` as image extension; use input image suffix or fallback `.png`).
	- Introduce multi-branch orchestration primitives and extend pipeline models to record per-branch artifacts and metrics.
	- Update artifact layout: `original/`, `generated/`, `result/`, `artifacts/`, and centralized `report.json`.
	- Add `AGENTS.md` to document agent execution contracts and observability requirements.

These changes were applied to pipeline modules and documentation to make end-to-end runs more robust and observable.

