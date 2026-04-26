from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageName(str, Enum):
    RESEARCH = "research"
    CODEGEN = "codegen"
    OPTIMIZER = "optimizer"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    PACKAGE = "package"


@dataclass(slots=True)
class PipelineConfig:
    output_root: Path = Path("output")
    algorithms_root: Path | None = None
    session_id: str | None = None
    timeout_seconds: int = 180
    llm_max_retries: int = 3
    http_max_retries: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 8.0
    retry_jitter: float = 0.2
    llm_timeout_seconds: float = 45.0
    http_timeout_seconds: float = 30.0
    executor_retry_once: bool = False
    optimizer_module: str | None = None
    optimizer_function: str = "optimize"
    continue_on_optimizer_failure: bool = False
    max_memory_mb: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        if self.llm_max_retries < 0:
            raise ValueError("llm_max_retries must be >= 0")

        if self.http_max_retries < 0:
            raise ValueError("http_max_retries must be >= 0")

        if self.retry_initial_delay < 0:
            raise ValueError("retry_initial_delay must be >= 0")

        if self.retry_max_delay < 0:
            raise ValueError("retry_max_delay must be >= 0")

        if self.retry_jitter < 0:
            raise ValueError("retry_jitter must be >= 0")

        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be greater than 0")

        if self.http_timeout_seconds <= 0:
            raise ValueError("http_timeout_seconds must be greater than 0")

        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be greater than 0 when provided")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["algorithms_root"] = str(
            self.algorithms_root) if self.algorithms_root else None
        return data


@dataclass(slots=True)
class PipelineRequest:
    image_path: Path
    scene_prompt: str
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "scene_prompt": self.scene_prompt,
            "config": self.config.to_dict(),
        }


@dataclass(slots=True)
class CandidateMethod:
    name: str
    description: str
    parameters: dict[str, Any]
    rationale: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(slots=True)
class ResearchResult:
    scene_prompt: str
    candidates: list[CandidateMethod]
    chosen_strategy: str
    evaluation_metrics: list[str]
    evaluation_plan: str
    summary: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_prompt": self.scene_prompt,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "chosen_strategy": self.chosen_strategy,
            "evaluation_metrics": self.evaluation_metrics,
            "evaluation_plan": self.evaluation_plan,
            "summary": self.summary,
            "sources": self.sources,
        }


@dataclass(slots=True)
class GeneratedAlgorithmArtifact:
    path: Path
    prepare_path: Path | None
    source_hash: str
    strategy_name: str
    syntax_validated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "prepare_path": str(self.prepare_path) if self.prepare_path else None,
            "source_hash": self.source_hash,
            "strategy_name": self.strategy_name,
            "syntax_validated": self.syntax_validated,
        }


@dataclass(slots=True)
class ExecutionResult:
    success: bool
    command: list[str]
    returncode: int | None
    output_image_path: Path | None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "returncode": self.returncode,
            "output_image_path": str(self.output_image_path) if self.output_image_path else None,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "timed_out": self.timed_out,
        }


@dataclass(slots=True)
class QualityReport:
    psnr: float | None
    ssim: float | None
    latency_seconds: float
    score: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StageRecord:
    name: StageName
    status: StageStatus = StageStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    error_code: str | None = None
    message: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_code": self.error_code,
            "message": self.message,
            "artifacts": self.artifacts,
        }


@dataclass(slots=True)
class PipelineResult:
    run_id: str
    request: PipelineRequest
    research_result: ResearchResult | None = None
    algorithm_artifact: GeneratedAlgorithmArtifact | None = None
    execution_result: ExecutionResult | None = None
    quality_report: QualityReport | None = None
    stage_records: list[StageRecord] = field(default_factory=list)
    manifest_path: Path | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return self.error_code is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "request": self.request.to_dict(),
            "research_result": self.research_result.to_dict() if self.research_result else None,
            "algorithm_artifact": self.algorithm_artifact.to_dict() if self.algorithm_artifact else None,
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "stage_records": [record.to_dict() for record in self.stage_records],
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "success": self.success,
        }
