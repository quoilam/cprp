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

