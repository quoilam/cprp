from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("output")
DEFAULT_SESSION_ID: str | None = None
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_LLM_MAX_RETRIES = 3
DEFAULT_HTTP_MAX_RETRIES = 3
DEFAULT_RETRY_INITIAL_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 8.0
DEFAULT_RETRY_JITTER = 0.2
DEFAULT_LLM_TIMEOUT_SECONDS = 45.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_EXECUTOR_RETRY_ONCE = False
DEFAULT_OPTIMIZER_MODULE = "optimizers.autoresearch"
DEFAULT_OPTIMIZER_FUNCTION = "optimize"
DEFAULT_CONTINUE_ON_OPTIMIZER_FAILURE = False
DEFAULT_MAX_MEMORY_MB: int | None = None

# Multi-branch pipeline configuration
DEFAULT_MAX_BRANCHES = 2  # Maximum number of candidates to generate code for
DEFAULT_BYPASS_AUTORESEARCH = True  # Bypass the time-consuming autoresearch by default
DEFAULT_SELECTION_STRATEGY = "best_score"  # How to select among branches: "best_score" | "highest_confidence" | "first_success"


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
    """Configuration for the pipeline behavior."""
    max_branches: int = DEFAULT_MAX_BRANCHES
    bypass_autoresearch: bool = DEFAULT_BYPASS_AUTORESEARCH
    selection_strategy: str = DEFAULT_SELECTION_STRATEGY  # "best_score" | "highest_confidence" | "first_success"
    optimizer_module: str = DEFAULT_OPTIMIZER_MODULE
    optimizer_function: str = DEFAULT_OPTIMIZER_FUNCTION
    continue_on_optimizer_failure: bool = DEFAULT_CONTINUE_ON_OPTIMIZER_FAILURE
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    http_max_retries: int = DEFAULT_HTTP_MAX_RETRIES
    retry_initial_delay: float = DEFAULT_RETRY_INITIAL_DELAY
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY
    retry_jitter: float = DEFAULT_RETRY_JITTER
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS
    executor_timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    executor_retry_once: bool = DEFAULT_EXECUTOR_RETRY_ONCE
    max_memory_mb: int | None = DEFAULT_MAX_MEMORY_MB

    @classmethod
    def from_args(cls, **overrides: Any) -> "PipelineConfig":
        """Create a config, overriding specific fields from kwargs."""
        fields = {
            "max_branches", "bypass_autoresearch", "selection_strategy",
            "optimizer_module", "optimizer_function", "continue_on_optimizer_failure",
            "llm_max_retries", "http_max_retries",
            "retry_initial_delay", "retry_max_delay", "retry_jitter",
            "llm_timeout_seconds", "http_timeout_seconds",
            "executor_timeout_seconds", "executor_retry_once", "max_memory_mb",
        }
        kwargs = {}
        for f in fields:
            if f in overrides:
                kwargs[f] = overrides[f]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_branches": self.max_branches,
            "bypass_autoresearch": self.bypass_autoresearch,
            "selection_strategy": self.selection_strategy,
            "optimizer_module": self.optimizer_module,
            "optimizer_function": self.optimizer_function,
            "continue_on_optimizer_failure": self.continue_on_optimizer_failure,
            "llm_max_retries": self.llm_max_retries,
            "http_max_retries": self.http_max_retries,
            "retry_initial_delay": self.retry_initial_delay,
            "retry_max_delay": self.retry_max_delay,
            "retry_jitter": self.retry_jitter,
            "llm_timeout_seconds": self.llm_timeout_seconds,
            "http_timeout_seconds": self.http_timeout_seconds,
            "executor_timeout_seconds": self.executor_timeout_seconds,
            "executor_retry_once": self.executor_retry_once,
            "max_memory_mb": self.max_memory_mb,
        }


@dataclass(slots=True)
class PipelineRequest:
    image_path: Path
    scene_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "scene_prompt": self.scene_prompt,
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
    # Multi-branch tracking: which candidate from research this code was generated from
    candidate_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "prepare_path": str(self.prepare_path) if self.prepare_path else None,
            "source_hash": self.source_hash,
            "strategy_name": self.strategy_name,
            "syntax_validated": self.syntax_validated,
            "candidate_index": self.candidate_index,
        }


@dataclass(slots=True)
class OptimizedArtifact:
    """Tracks an optimized version of an algorithm artifact."""
    original_path: Path
    optimized_path: Path
    strategy_name: str
    candidate_index: int = 0
    changed: bool = False
    snapshot_path: Path | None = None
    # Store before/after hashes for comparison
    before_hash: str = ""
    after_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_path": str(self.original_path),
            "optimized_path": str(self.optimized_path),
            "strategy_name": self.strategy_name,
            "candidate_index": self.candidate_index,
            "changed": self.changed,
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path else None,
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
        }


@dataclass(slots=True)
class CodegenBranchResult:
    """Result of a single code generation branch."""
    candidate: CandidateMethod
    artifact: GeneratedAlgorithmArtifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": {
                "name": self.candidate.name,
                "confidence": self.candidate.confidence,
                "description": self.candidate.description,
            },
            "artifact": self.artifact.to_dict(),
        }


@dataclass(slots=True)
class ExecutionBranchResult:
    """Execution result with branch context."""
    execution_result: ExecutionResult
    strategy_name: str
    candidate_index: int
    optimized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_result": self.execution_result.to_dict(),
            "strategy_name": self.strategy_name,
            "candidate_index": self.candidate_index,
            "optimized": self.optimized,
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
    # Single-branch (legacy) support
    algorithm_artifact: GeneratedAlgorithmArtifact | None = None
    execution_result: ExecutionResult | None = None
    quality_report: QualityReport | None = None
    # Multi-branch support
    codegen_branches: list[CodegenBranchResult] = field(default_factory=list)
    optimized_branches: list[OptimizedArtifact] = field(default_factory=list)
    execution_branches: list[ExecutionBranchResult] = field(default_factory=list)
    quality_branches: list[tuple[int, QualityReport]] = field(default_factory=list)  # (candidate_index, report)
    # Final selected branch info
    selected_branch_index: int | None = None
    selected_strategy_name: str | None = None
    selected_output_image: Path | None = None
    # General
    stage_records: list[StageRecord] = field(default_factory=list)
    manifest_path: Path | None = None
    error_code: str | None = None
    error_message: str | None = None
    # Pipeline configuration snapshot
    config: PipelineConfig | None = None

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
            "codegen_branches": [b.to_dict() for b in self.codegen_branches],
            "optimized_branches": [b.to_dict() for b in self.optimized_branches],
            "execution_branches": [b.to_dict() for b in self.execution_branches],
            "quality_branches": [
                {"candidate_index": idx, "report": report.to_dict()}
                for idx, report in self.quality_branches
            ],
            "selected_branch_index": self.selected_branch_index,
            "selected_strategy_name": self.selected_strategy_name,
            "selected_output_image": str(self.selected_output_image) if self.selected_output_image else None,
            "stage_records": [record.to_dict() for record in self.stage_records],
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "config": self.config.to_dict() if self.config else None,
            "success": self.success,
        }


@dataclass(slots=True)
class BranchComparisonReport:
    """Report comparing all branches."""
    branches: list[dict[str, Any]] = field(default_factory=list)
    selection_reason: str = ""
    selection_criteria: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": self.branches,
            "selection_reason": self.selection_reason,
            "selection_criteria": self.selection_criteria,
        }
