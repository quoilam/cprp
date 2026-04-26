from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import shutil
import uuid

from .models import DEFAULT_OUTPUT_ROOT, DEFAULT_SESSION_ID, PipelineRequest, StageName, StageRecord, StageStatus


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


def _update_latest_symlink(root: Path, session_dir: Path) -> None:
    latest_path = root / "latest"
    target_name = session_dir.name

    if latest_path.is_symlink() or latest_path.is_file():
        latest_path.unlink()
    elif latest_path.exists():
        return
    latest_path.symlink_to(target_name, target_is_directory=True)


# ---------- 中文阶段名称映射 ----------
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
    _stage_start_times: dict[StageName, float] = field(
        default_factory=dict, init=False)

    @classmethod
    def create(cls, request: PipelineRequest) -> "PipelineContext":
        session_id = DEFAULT_SESSION_ID or make_session_id()
        run_id = make_run_id(request.scene_prompt)
        paths = PipelinePaths.create(DEFAULT_OUTPUT_ROOT, session_id, run_id)
        context = cls(request=request, session_id=session_id,
                      run_id=run_id, paths=paths)
        context.metadata["created_at"] = _utc_now()
        context.metadata["session_id"] = session_id
        # 仅写入日志文件，不打印到控制台（由美化方法接管）
        context._write_event_log("pipeline", "created", {
                                 "session_id": session_id, "run_id": run_id})
        return context

    # ---------- 内部日志写入（保留原功能，供审计） ----------
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

    # 保留原 log_event 方法，但控制台打印改为调用美化方法（稍后重定向）
    def log_event(self, category: str, event: str, payload: dict[str, Any] | None = None) -> Path:
        # 写入日志文件
        path = self._write_event_log(category, event, payload)
        # 控制台输出由专门的美化方法处理，这里不再打印原始格式
        # 为了兼容性，保留该方法，但实际打印已在 runner/stages 中通过美化方法实现
        return path

    # ---------- 美化控制台输出方法 ----------
    def print_header(self) -> None:
        """打印流水线启动信息"""
        print("\n" + "=" * 70)
        print("🚀 流水线启动")
        print(f"   会话ID: {self.session_id}")
        print(f"   运行ID: {self.run_id}")
        print(f"   输入图片: {self.request.image_path}")
        print(
            f"   场景提示: {self.request.scene_prompt[:60]}{'...' if len(self.request.scene_prompt) > 60 else ''}")
        print("=" * 70)

    def print_stage_start(self, stage: StageName, message: str = "") -> None:
        """打印阶段开始信息"""
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
        """打印阶段结束信息（成功/失败/跳过）"""
        import time
        start_time = self._stage_start_times.get(stage)
        duration = ""
        if start_time:
            elapsed = time.time() - start_time
            duration = f" (耗时 {elapsed:.1f}秒)"

        cn_name = _STAGE_NAMES_CN.get(stage, stage.value)
        if status == StageStatus.SUCCEEDED:
            icon = "✅"
            status_text = "成功"
        elif status == StageStatus.FAILED:
            icon = "❌"
            status_text = "失败"
        elif status == StageStatus.SKIPPED:
            icon = "⏭️"
            status_text = "已跳过"
        else:
            icon = "⏳"
            status_text = status.value

        print(f"   {icon} {cn_name}阶段{status_text}{duration}")
        if message:
            print(f"      {message}")
        if extra:
            for key, val in extra.items():
                print(f"      {key}: {val}")

    def print_footer(self, success: bool, output_paths: dict[str, Any] | None = None) -> None:
        """打印流水线结束汇总"""
        print("\n" + "=" * 70)
        if success:
            print("✅ 流水线执行成功")
        else:
            print("❌ 流水线执行失败")
        if output_paths:
            print(f"   输出图片: {output_paths.get('output_image', 'N/A')}")
            print(f"   产物清单: {output_paths.get('manifest', 'N/A')}")
        print("=" * 70 + "\n")

    # ---------- 阶段管理方法（内部仍写入JSON记录，同时调用美化打印） ----------
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
        self._write_event_log(
            "stage", "start", {"stage": stage_name.value, "message": message})
        # 美化打印
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
        self.write_stage_record(stage_name)
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
        # 美化打印
        self.print_stage_result(stage_name, status, message or "", extra_info)
        return record

    def write_stage_record(self, stage_name: StageName) -> Path:
        record = self.ensure_stage_record(stage_name)
        stage_path = self.paths.stages_dir / f"{stage_name.value}.json"
        stage_path.write_text(json.dumps(
            record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return stage_path

    def copy_input_image(self) -> Path:
        source = self.request.image_path
        if not source.exists():
            raise FileNotFoundError(f"image_path not found: {source}")
        destination = self.paths.inputs_dir / source.name
        shutil.copy2(source, destination)
        self.artifacts["input_image"] = destination
        self._write_event_log("artifact", "copied_input", {
                              "source": source, "destination": destination})
        return destination

    def write_json(self, relative_name: str, payload: dict[str, Any]) -> Path:
        destination = self.paths.run_dir / relative_name
        destination.write_text(json.dumps(
            payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_event_log("artifact", "write_json", {
                              "path": destination, "name": relative_name})
        return destination
