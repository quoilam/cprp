# Development Notes

## Architecture

The project is being collapsed into a single pipeline entrypoint. The CLI accepts only image-related inputs and produces a run directory with all artifacts, stage manifests, and metrics.

## Current Contracts

- `PipelineRequest` carries the image path, scene prompt, and runtime config.
- `PipelineContext` owns the run ID, artifact paths, and stage records.
- Generated algorithm files must expose `run(image_path: str, output_path: str, scene_prompt: str) -> dict`.
- The optimizer protocol is fixed as `optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None`.

## Artifact Layout

- `output/<session_id>/<run_id>/inputs/` stores copied input assets.
- `output/<session_id>/<run_id>/artifacts/` stores snapshots and stage byproducts.
- `output/<session_id>/<run_id>/stages/` stores per-stage JSON state.
- `output/<session_id>/<run_id>/manifest.json` stores the final result package.
- `output/latest` is a soft link pointing to the latest `<session_id>` directory.

## Verification Targets

1. One command should run research, code generation, optimizer, executor, evaluator, and packaging.
2. Every stage should record status and errors.
3. The executor must run the algorithm file in a subprocess and enforce a timeout.
4. The evaluator should emit PSNR, SSIM, latency, and a combined score.

