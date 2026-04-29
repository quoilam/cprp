import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipeline import PipelineConfig, PipelineRequest, PipelineRunner


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Image algorithm automation pipeline")
    parser.add_argument("--image-path", required=True,
                        help="Path to the input image")
    parser.add_argument("--scene-prompt", required=True,
                        help="Scene prompt describing the target task")

    parser.add_argument("--bypass-autoresearch", action="store_true", default=True,
                        help="Bypass the time-consuming autoresearch agent (default: True)")
    parser.add_argument("--no-bypass-autoresearch", action="store_false", dest="bypass_autoresearch",
                        help="Do NOT bypass autoresearch (will call external agent)")
    parser.add_argument("--continue-on-optimizer-failure", action="store_true",
                        help="Continue pipeline even if optimizer fails (default: False)")

    args = parser.parse_args()

    config = PipelineConfig.from_args(
        bypass_autoresearch=args.bypass_autoresearch,
        continue_on_optimizer_failure=args.continue_on_optimizer_failure,
    )

    request = PipelineRequest(
        image_path=Path(args.image_path),
        scene_prompt=args.scene_prompt,
    )

    runner = PipelineRunner(config=config)
    result = runner.run(request)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result.success else 1)
