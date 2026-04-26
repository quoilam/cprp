import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipeline import PipelineConfig, PipelineRequest, PipelineRunner


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Image algorithm automation pipeline")
    parser.add_argument("--image-path", required=True, help="Path to the input image")
    parser.add_argument("--scene-prompt", required=True, help="Scene prompt describing the target task")
    parser.add_argument("--output-root", default="output", help="Root directory for session artifacts")
    parser.add_argument("--session-id", default=None, help="Session identifier used to group runs")
    parser.add_argument("--algorithms-root", default=None, help="Optional external directory for generated algorithm files")
    parser.add_argument("--timeout-seconds", type=_positive_int, default=180, help="Execution timeout in seconds")
    parser.add_argument("--max-memory-mb", type=_positive_int, default=None, help="Optional memory limit for the executor")
    parser.add_argument("--llm-max-retries", type=_non_negative_int, default=3, help="Max retry attempts for LLM requests")
    parser.add_argument("--http-max-retries", type=_non_negative_int, default=3, help="Max retry attempts for HTTP requests")
    parser.add_argument("--retry-initial-delay", type=_non_negative_float, default=1.0, help="Initial backoff delay in seconds")
    parser.add_argument("--retry-max-delay", type=_non_negative_float, default=8.0, help="Maximum backoff delay in seconds")
    parser.add_argument("--retry-jitter", type=_non_negative_float, default=0.2, help="Random jitter added to each retry delay")
    parser.add_argument("--llm-timeout-seconds", type=_positive_float, default=45.0, help="Timeout for LLM calls in seconds")
    parser.add_argument("--http-timeout-seconds", type=_positive_float, default=30.0, help="Timeout for HTTP calls in seconds")
    parser.add_argument("--executor-retry-once", action="store_true", help="Retry executor subprocess once when enabled")
    parser.add_argument("--optimizer-module", default="optimizers.autoresearch", help="Optional optimizer module name")
    parser.add_argument("--optimizer-function", default="optimize", help="Optimizer function name")
    parser.add_argument("--continue-on-optimizer-failure", action="store_true", help="Continue when optimization fails")

    args = parser.parse_args()

    request = PipelineRequest(
        image_path=Path(args.image_path),
        scene_prompt=args.scene_prompt,
        config=PipelineConfig(
            output_root=Path(args.output_root),
            algorithms_root=Path(
                args.algorithms_root) if args.algorithms_root else None,
            session_id=args.session_id,
            timeout_seconds=args.timeout_seconds,
            llm_max_retries=args.llm_max_retries,
            http_max_retries=args.http_max_retries,
            retry_initial_delay=args.retry_initial_delay,
            retry_max_delay=args.retry_max_delay,
            retry_jitter=args.retry_jitter,
            llm_timeout_seconds=args.llm_timeout_seconds,
            http_timeout_seconds=args.http_timeout_seconds,
            executor_retry_once=args.executor_retry_once,
            optimizer_module=args.optimizer_module,
            optimizer_function=args.optimizer_function,
            continue_on_optimizer_failure=args.continue_on_optimizer_failure,
            max_memory_mb=args.max_memory_mb,
        ),
    )

    result = PipelineRunner().run(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result.success else 1)
