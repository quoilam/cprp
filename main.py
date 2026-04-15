import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipeline import PipelineConfig, PipelineRequest, PipelineRunner


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Image algorithm automation pipeline")
    parser.add_argument("--image-path", required=True, help="Path to the input image")
    parser.add_argument("--scene-prompt", required=True, help="Scene prompt describing the target task")
    parser.add_argument("--output-root", default="output", help="Root directory for session artifacts")
    parser.add_argument("--session-id", default=None, help="Session identifier used to group runs")
    parser.add_argument("--algorithms-root", default=None, help="Optional external directory for generated algorithm files")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Execution timeout in seconds")
    parser.add_argument("--max-memory-mb", type=int, default=None, help="Optional memory limit for the executor")
    parser.add_argument("--optimizer-module", default=None, help="Optional optimizer module name")
    parser.add_argument("--optimizer-function", default="optimize", help="Optimizer function name")
    parser.add_argument("--continue-on-optimizer-failure", action="store_true", help="Continue when optimization fails")
    args = parser.parse_args()

    request = PipelineRequest(
        image_path=Path(args.image_path),
        scene_prompt=args.scene_prompt,
        config=PipelineConfig(
            output_root=Path(args.output_root),
            algorithms_root=Path(args.algorithms_root) if args.algorithms_root else None,
            session_id=args.session_id,
            timeout_seconds=args.timeout_seconds,
            optimizer_module=args.optimizer_module,
            optimizer_function=args.optimizer_function,
            continue_on_optimizer_failure=args.continue_on_optimizer_failure,
            max_memory_mb=args.max_memory_mb,
        ),
    )

    result = PipelineRunner().run(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result.success else 1)
