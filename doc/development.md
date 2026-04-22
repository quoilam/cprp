# Development Notes

## Architecture

The project now runs as a single image-focused pipeline entrypoint (`main.py`) backed by `PipelineRunner`.

- Input: one image path + one scene prompt.
- Output: one run directory containing generated algorithm code, stage states, logs, metrics, and a manifest.
- Orchestration order is fixed: `research -> codegen -> optimizer -> executor -> evaluator -> package`.
- Even when a stage fails, the pipeline still writes a final manifest in `package` stage.

## CLI Contract (`main.py`)

Required arguments:

- `--image-path`
- `--scene-prompt`

Optional runtime arguments:

- `--output-root` (default: `output`)
- `--session-id` (default: auto-generated `session_<10hex>`)
- `--algorithms-root` (default: run-local `algorithms/`)
- `--timeout-seconds` (default: `180`)
- `--max-memory-mb` (optional executor memory cap on POSIX)
- `--optimizer-module` (optional, dynamic import)
- `--optimizer-function` (default: `optimize`)
- `--continue-on-optimizer-failure` (default: false)

Exit code:

- `0` when `PipelineResult.success` is true.
- `1` when any pipeline error is recorded.

## Core Data Contracts

- `PipelineRequest(image_path, scene_prompt, config)` is the run input envelope.
- `PipelineContext` owns `session_id`, `run_id`, all paths, stage records, artifact pointers, and structured event logs.
- Generated algorithm file contract:
  `run(image_path: str, output_path: str, scene_prompt: str) -> dict`
- Optimizer protocol contract:
  `optimize(algo_file_path: str, user_prompt: str, user_image_file_path: str) -> None`

Stage names (`StageName`):

- `research`
- `codegen`
- `optimizer`
- `executor`
- `evaluator`
- `package`

Stage statuses (`StageStatus`):

- `pending`
- `running`
- `succeeded`
- `failed`
- `skipped`

## Stage Behavior (Current Implementation)

1. `research`
	- Optional Tavily web clues when `TAVILY_API_KEY` is present.
	- OpenRouter JSON response parsing into `ResearchResult`.
	- Writes `research.json` and `research_web_clues.json`.

2. `codegen`
	- Uses OpenRouter to generate Python algorithm source.
	- Validates syntax, required `run(...)` signature, and allowed imports.
	- Runs contract verification by subprocess CLI invocation.
	- Retries up to 3 attempts when verification fails.
	- Writes `codegen.json`.

3. `optimizer`
	- If `optimizer_module` is unset: treated as successful no-op.
	- If set: imports and calls configured optimizer function.
	- Detects in-place code changes via file hash.
	- Saves optimized snapshot to `artifacts/*.optimized.py` when changed.

4. `executor`
	- Runs generated algorithm via subprocess (`sys.executable`).
	- Passes `--image-path --output-path --scene-prompt`.
	- Enforces timeout (`timeout_seconds`).
	- Applies optional memory limit (`RLIMIT_AS`) on POSIX.
	- Writes `execution.json`.

5. `evaluator`
	- Computes PSNR and SSIM between input image and output image.
	- Computes latency score from execution time.
	- Produces combined score and writes `quality.json`.

6. `package`
	- Writes `manifest.json` from `PipelineResult.to_dict()`.
	- Executed in both success and failure paths.

## Artifact Layout (Current)

Per run:

- `output/<session_id>/<run_id>/algorithms/`
- `output/<session_id>/<run_id>/inputs/`
- `output/<session_id>/<run_id>/artifacts/`
- `output/<session_id>/<run_id>/stages/`
- `output/<session_id>/<run_id>/logs/events.jsonl`
- `output/<session_id>/<run_id>/output/`
- `output/<session_id>/<run_id>/research.json`
- `output/<session_id>/<run_id>/research_web_clues.json`
- `output/<session_id>/<run_id>/codegen.json`
- `output/<session_id>/<run_id>/execution.json`
- `output/<session_id>/<run_id>/quality.json`
- `output/<session_id>/<run_id>/manifest.json`

Session pointer:

- `output/latest` is maintained as a symlink to the latest `<session_id>` directory.
- If `output/latest` already exists as a real directory, it is intentionally left untouched.

## Evaluation Formula (Current)

- `normalized_psnr = min(psnr / 40, 1)`
- `latency_score = max(0, 1 - duration_seconds / timeout_seconds)`
- `score = mean([normalized_psnr, max(0, ssim), latency_score])`

Quality report fields:

- `psnr`
- `ssim`
- `latency_seconds`
- `score`
- `notes`

## External Dependencies and Environment

For research/codegen:

- `OPENROUTER_API_KEY` (required)
- `OPENROUTER_MODEL` (required)
- `OPENROUTER_BASE_URL` (optional, defaults to OpenRouter API)

For web clues (optional):

- `TAVILY_API_KEY`

If OpenRouter env vars are missing, research/codegen stages will fail and the run will still be packaged with failure details in stage records and manifest.

