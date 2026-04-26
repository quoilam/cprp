from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import re
import shutil
import uuid

from .models import PipelineRequest, StageName, StageRecord, StageStatus


# ---------- 工具函数 ----------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def make_run_id(scene_prompt: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{timestamp}_{_short_hash(scene_prompt + str(uuid.uuid4()))}"


def make_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:10]}"


def _make_run_dir_name(scene_prompt: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _update_latest_symlink(root: Path, target_dir: Path) -> None:
    latest_path = root / "latest"
    target_name = target_dir.name
    if latest_path.is_symlink() or latest_path.is_file():
        latest_path.unlink()
    elif latest_path.exists():
        return
    latest_path.symlink_to(target_name, target_is_directory=True)


# 中文阶段名称映射
_STAGE_NAMES_CN = {
    StageName.RESEARCH: "研究",
    StageName.CODEGEN: "代码生成",
    StageName.OPTIMIZER: "优化",
    StageName.EXECUTOR: "执行",
    StageName.EVALUATOR: "评估",
    StageName.PACKAGE: "打包",
}

_STAGE_ORDER = [
    StageName.RESEARCH,
    StageName.CODEGEN,
    StageName.OPTIMIZER,
    StageName.EXECUTOR,
    StageName.EVALUATOR,
    StageName.PACKAGE,
]


@dataclass(slots=True)
class PipelinePaths:
    """扁平化输出目录结构"""
    root: Path
    run_dir: Path
    original_dir: Path
    generated_dir: Path
    result_dir: Path
    events_log_path: Path
    report_path: Path

    @classmethod
    def create(cls, output_root: Path, scene_prompt: str) -> "PipelinePaths":
        absolute_root = output_root.resolve()
        run_dir_name = _make_run_dir_name(scene_prompt)
        run_dir = absolute_root / run_dir_name

        paths = cls(
            root=absolute_root,
            run_dir=run_dir,
            original_dir=run_dir / "original",
            generated_dir=run_dir / "generated",
            result_dir=run_dir / "result",
            events_log_path=run_dir / "events.jsonl",
            report_path=run_dir / "report.json",
        )

        for directory in [paths.original_dir, paths.generated_dir, paths.result_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        _update_latest_symlink(paths.root, run_dir)
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
    _stage_start_times: dict[StageName, float] = field(default_factory=dict, init=False)

    @classmethod
    def create(cls, request: PipelineRequest) -> "PipelineContext":
        session_id = request.config.session_id or make_session_id()
        run_id = make_run_id(request.scene_prompt)
        paths = PipelinePaths.create(request.config.output_root, request.scene_prompt)

        context = cls(
            request=request,
            session_id=session_id,
            run_id=run_id,
            paths=paths,
        )
        context.metadata["created_at"] = _utc_now()
        context.metadata["session_id"] = session_id
        context._write_event_log("pipeline", "created", {"session_id": session_id, "run_id": run_id})
        return context

    # ---------- 内部日志 ----------
    def _write_event_log(self, category: str, event: str, payload: dict[str, Any] | None = None) -> Path:
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
        return self.paths.events_log_path

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._to_json_safe(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(item) for item in value]
        return value

    def log_event(self, category: str, event: str, payload: dict[str, Any] | None = None) -> Path:
        return self._write_event_log(category, event, payload)

    # ---------- 控制台美化输出 ----------
    def print_header(self) -> None:
        print("\n" + "=" * 70)
        print("🚀 流水线启动")
        print(f"   会话ID: {self.session_id}")
        print(f"   运行ID: {self.run_id}")
        print(f"   输入图片: {self.request.image_path}")
        prompt_preview = self.request.scene_prompt[:60]
        if len(self.request.scene_prompt) > 60:
            prompt_preview += "..."
        print(f"   场景提示: {prompt_preview}")
        print("=" * 70)

    def print_stage_start(self, stage: StageName, message: str = "") -> None:
        import time
        self._stage_start_times[stage] = time.time()
        index = _STAGE_ORDER.index(stage) + 1
        total = len(_STAGE_ORDER)
        cn_name = _STAGE_NAMES_CN.get(stage, stage.value)
        print(f"\n🔹 [{index}/{total}] {cn_name}阶段")
        if message:
            print(f"   → {message}")

    def print_stage_result(
        self,
        stage: StageName,
        status: StageStatus,
        message: str = "",
        extra: dict[str, Any] | None = None
    ) -> None:
        import time
        start_time = self._stage_start_times.get(stage)
        duration = ""
        if start_time:
            elapsed = time.time() - start_time
            duration = f" (耗时 {elapsed:.1f}秒)"

        cn_name = _STAGE_NAMES_CN.get(stage, stage.value)
        if status == StageStatus.SUCCEEDED:
            icon, status_text = "✅", "成功"
        elif status == StageStatus.FAILED:
            icon, status_text = "❌", "失败"
        elif status == StageStatus.SKIPPED:
            icon, status_text = "⏭️", "已跳过"
        else:
            icon, status_text = "⏳", status.value

        print(f"   {icon} {cn_name}阶段{status_text}{duration}")
        if message:
            print(f"      {message}")
        if extra:
            for key, val in extra.items():
                print(f"      {key}: {val}")

    def print_footer(self, success: bool, output_paths: dict[str, Any] | None = None) -> None:
        print("\n" + "=" * 70)
        if success:
            print("✅ 流水线执行成功")
        else:
            print("❌ 流水线执行失败")
        if output_paths:
            print(f"   输出图片: {output_paths.get('output_image', 'N/A')}")
            print(f"   报告文件: {output_paths.get('report', 'N/A')}")
        print("=" * 70 + "\n")

    # ---------- 阶段管理 ----------
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
        self._write_event_log("stage", "start", {"stage": stage_name.value, "message": message})
        self.print_stage_start(stage_name, message or "")
        return record

    def finish_stage(
        self,
        stage_name: StageName,
        status: StageStatus,
        message: str | None = None,
        error_code: str | None = None,
        artifacts: list[Path] | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> StageRecord:
        record = self.ensure_stage_record(stage_name)
        record.status = status
        record.finished_at = _utc_now()
        record.message = message
        record.error_code = error_code
        if artifacts:
            record.artifacts.extend(str(path) for path in artifacts)
        self._write_event_log(
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
        self.print_stage_result(stage_name, status, message or "", extra_info)
        return record

    def copy_input_image(self) -> Path:
        source = self.request.image_path
        if not source.exists():
            raise FileNotFoundError(f"image_path not found: {source}")
        destination = self.paths.original_dir / source.name
        shutil.copy2(source, destination)
        self.artifacts["input_image"] = destination
        self._write_event_log("artifact", "copied_input", {"source": source, "destination": destination})
        return destination