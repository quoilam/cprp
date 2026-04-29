from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.context import PipelineContext


def package_stage(context: PipelineContext, payload: dict[str, Any]) -> Path:
    manifest_path = context.paths.manifest_path
    manifest_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
