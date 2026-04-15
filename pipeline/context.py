from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import shutil
import uuid

from .models import PipelineRequest, StageName, StageRecord, StageStatus


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def make_run_id(scene_prompt: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{_short_hash(scene_prompt + str(uuid.uuid4()))}"


def make_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:10]}"


def _update_latest_symlink(root: Path, session_dir: Path) -> None:
    latest_path = root / "latest"
    target_name = session_dir.name

    if latest_path.is_symlink() or latest_path.is_file():
        latest_path.unlink()
    elif latest_path.exists():
        # Keep a real directory untouched to avoid destructive behavior.
        return

    latest_path.symlink_to(target_name, target_is_directory=True)


@dataclass(slots=True)
class PipelinePaths:
    root: Path
    session_dir: Path
    run_dir: Path
    generated_algorithms_dir: Path
    inputs_dir: Path
    artifacts_dir: Path
    stages_dir: Path
    logs_dir: Path
    events_log_path: Path
    output_dir: Path
    manifest_path: Path

    @classmethod
    def create(cls, output_root: Path, session_id: str, run_id: str) -> "PipelinePaths":
        absolute_root = output_root.resolve()
        session_dir = absolute_root / session_id
        run_dir = session_dir / run_id
        paths = cls(
            root=absolute_root,
            session_dir=session_dir,
            run_dir=run_dir,
            generated_algorithms_dir=run_dir / "algorithms",
            inputs_dir=run_dir / "inputs",
            artifacts_dir=run_dir / "artifacts",
            stages_dir=run_dir / "stages",
            logs_dir=run_dir / "logs",
            events_log_path=run_dir / "logs" / "events.jsonl",
            output_dir=run_dir / "output",
            manifest_path=run_dir / "manifest.json",
        )
        for directory in [
            paths.root,
            paths.session_dir,
            paths.run_dir,
            paths.generated_algorithms_dir,
            paths.inputs_dir,
            paths.artifacts_dir,
            paths.stages_dir,
            paths.logs_dir,
            paths.output_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        _update_latest_symlink(paths.root, paths.session_dir)
        return paths


@dataclass(slots=True)
class PipelineContext:
    request: PipelineRequest
    session_id: str
    run_id: str
    paths: PipelinePaths
    stage_records: dict[StageName, StageRecord] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, request: PipelineRequest) -> "PipelineContext":
        session_id = request.config.session_id or make_session_id()
        run_id = make_run_id(request.scene_prompt)
        paths = PipelinePaths.create(request.config.output_root, session_id, run_id)
        context = cls(request=request, session_id=session_id, run_id=run_id, paths=paths)
        context.metadata["created_at"] = _utc_now()
        context.metadata["session_id"] = session_id
        context.log_event("pipeline", "created", {"session_id": session_id, "run_id": run_id})
        return context

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._to_json_safe(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(item) for item in value]
        return value

    def log_event(self, category: str, event: str, payload: dict[str, Any] | None = None) -> Path:
        record = {
            "ts": _utc_now(),
            "category": category,
            "event": event,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "payload": self._to_json_safe(payload or {}),
        }
        line = json.dumps(record, ensure_ascii=False)
        self.paths.events_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.events_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"[{category}] {event}: {json.dumps(record['payload'], ensure_ascii=False)}")
        return self.paths.events_log_path

    def ensure_stage_record(self, stage_name: StageName) -> StageRecord:
        record = self.stage_records.get(stage_name)
        if record is None:
            record = StageRecord(name=stage_name)
            self.stage_records[stage_name] = record
        return record

    def start_stage(self, stage_name: StageName, message: str | None = None) -> StageRecord:
        record = self.ensure_stage_record(stage_name)
        record.status = StageStatus.RUNNING
        record.started_at = _utc_now()
        record.message = message
        self.write_stage_record(stage_name)
        self.log_event("stage", "start", {"stage": stage_name.value, "message": message})
        return record

    def finish_stage(
        self,
        stage_name: StageName,
        status: StageStatus,
        message: str | None = None,
        error_code: str | None = None,
        artifacts: list[Path] | None = None,
    ) -> StageRecord:
        record = self.ensure_stage_record(stage_name)
        record.status = status
        record.finished_at = _utc_now()
        record.message = message
        record.error_code = error_code
        if artifacts:
            record.artifacts.extend(str(path) for path in artifacts)
        self.write_stage_record(stage_name)
        self.log_event(
            "stage",
            "finish",
            {
                "stage": stage_name.value,
                "status": status.value,
                "message": message,
                "error_code": error_code,
                "artifacts": record.artifacts,
            },
        )
        return record

    def write_stage_record(self, stage_name: StageName) -> Path:
        record = self.ensure_stage_record(stage_name)
        stage_path = self.paths.stages_dir / f"{stage_name.value}.json"
        stage_path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return stage_path

    def copy_input_image(self) -> Path:
        source = self.request.image_path
        if not source.exists():
            raise FileNotFoundError(f"image_path not found: {source}")
        destination = self.paths.inputs_dir / source.name
        shutil.copy2(source, destination)
        self.artifacts["input_image"] = destination
        self.log_event("artifact", "copied_input", {"source": source, "destination": destination})
        return destination

    def write_json(self, relative_name: str, payload: dict[str, Any]) -> Path:
        destination = self.paths.run_dir / relative_name
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log_event("artifact", "write_json", {"path": destination, "name": relative_name})
        return destination
