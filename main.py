import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from pipeline import PipelineRequest, PipelineRunner


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Image algorithm automation pipeline")
    parser.add_argument("--image-path", required=True,
                        help="Path to the input image")
    parser.add_argument("--scene-prompt", required=True,
                        help="Scene prompt describing the target task")

    args = parser.parse_args()

    request = PipelineRequest(
        image_path=Path(args.image_path),
        scene_prompt=args.scene_prompt,
    )

    result = PipelineRunner().run(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result.success else 1)
