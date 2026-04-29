from __future__ import annotations

import json
import os
from typing import Any

import requests

from prompts import RESEARCH_PROMPT_TEMPLATE
from pipeline.context import PipelineContext
from pipeline.models import (
    CandidateMethod,
    ResearchResult,
)
from pipeline.resilience import (
    BusinessLogicError,
    RetryableStatusCodeError,
    build_retry_decorator,
    build_retry_policy,
    extract_status_code,
)

from .common import call_openrouter_json


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _choose_strategy(scene_prompt: str) -> tuple[str, dict[str, Any], str]:
    normalized = _normalize_text(scene_prompt)
    crop_intent = any(keyword in normalized for keyword in [
        "crop", "center crop", "裁切", "裁剪", "居中", "中心", "提取"
    ])
    if crop_intent:
        crop_ratio = 0.5
        if any(keyword in normalized for keyword in ["30%", "0.3", "三成"]):
            crop_ratio = 0.3
        elif any(keyword in normalized for keyword in ["40%", "0.4", "四成"]):
            crop_ratio = 0.4
        elif any(keyword in normalized for keyword in ["50%", "0.5", "一半", "半", "half"]):
            crop_ratio = 0.5
        elif any(keyword in normalized for keyword in ["60%", "0.6", "六成"]):
            crop_ratio = 0.6
        elif any(keyword in normalized for keyword in ["70%", "0.7", "七成"]):
            crop_ratio = 0.7
        return "center_crop", {"crop_ratio": crop_ratio}, "Prompt suggests center crop extraction"
    if any(keyword in normalized for keyword in ["denoise", "noise", "去噪"]):
        return "denoise_conservative", {"filter_size": 3, "contrast": 1.03}, "Prompt suggests noise reduction"
    if any(keyword in normalized for keyword in ["deblur", "blur", "去模糊"]):
        return "deblur_sharpen", {"sharpness": 1.35, "detail": 1.08}, "Prompt suggests blur correction"
    if any(keyword in normalized for keyword in ["super-resolution", "upscale", "放大", "超分"]):
        return "upscale_refine", {"scale": 2, "sharpness": 1.1}, "Prompt suggests upscaling"
    if any(keyword in normalized for keyword in ["enhance", "contrast", "clear", "增强", "清晰"]):
        return "enhance_contrast", {"contrast": 1.15, "sharpness": 1.08}, "Prompt suggests global enhancement"
    return "balanced_enhancement", {"contrast": 1.08, "sharpness": 1.05}, "Default balanced image enhancement"


def _build_local_research_result(scene_prompt: str, web_clues: list[dict[str, str]]) -> ResearchResult:
    primary_strategy, primary_params, primary_rationale = _choose_strategy(
        scene_prompt)

    secondary_strategy = "enhance_contrast"
    secondary_params: dict[str, Any] = {"contrast": 1.12, "sharpness": 1.06}
    if primary_strategy == secondary_strategy:
        secondary_strategy = "denoise_conservative"
        secondary_params = {"filter_size": 3, "contrast": 1.03}

    candidates = [
        CandidateMethod(
            name=primary_strategy,
            description=f"Deterministic baseline selected from prompt intent: {primary_rationale}",
            parameters=primary_params,
            rationale=primary_rationale,
            sources=["local_heuristic"],
            confidence=0.82,
        ),
        CandidateMethod(
            name=secondary_strategy,
            description="Secondary robust fallback candidate.",
            parameters=secondary_params,
            rationale="Provide diversity as fallback candidate.",
            sources=["local_heuristic"],
            confidence=0.68,
        ),
    ]

    sources = [str(clue.get("url") or "")
               for clue in web_clues if str(clue.get("url") or "").strip()]
    return ResearchResult(
        scene_prompt=scene_prompt,
        candidates=candidates,
        chosen_strategy=candidates[0].name,
        evaluation_metrics=["psnr", "ssim", "latency"],
        evaluation_plan="Compute objective metrics on input/output pair and aggregate score.",
        summary="Built deterministic local research result with two candidate clues.",
        sources=sources,
    )


def _build_retry_policy(context: PipelineContext, *, max_retries: int | None = None):
    config = context.config
    return build_retry_policy(
        max_retries=config.http_max_retries if max_retries is None else max_retries,
        initial_delay=config.retry_initial_delay,
        max_delay=config.retry_max_delay,
        jitter=config.retry_jitter,
    )


def _search_web_clues(context: PipelineContext, scene_prompt: str) -> list[dict[str, str]]:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        context.log_event("tool_call", "tavily_skip", {
                          "reason": "TAVILY_API_KEY missing"})
        return []

    policy = _build_retry_policy(context)
    http_timeout = float(context.config.http_timeout_seconds)
    context.log_event(
        "tool_call",
        "tavily_start",
        {
            "query": f"image processing algorithm for: {scene_prompt}",
            "timeout_seconds": http_timeout,
            "max_retries": policy.max_retries,
            "max_attempts": policy.max_attempts,
        },
    )

    @build_retry_decorator(
        context=context,
        stage="research",
        operation="tavily_search",
        policy=policy,
    )
    def _request_once() -> list[dict[str, str]]:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_api_key,
                "query": f"image processing algorithm for: {scene_prompt}",
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": False,
                "include_images": False,
            },
            timeout=http_timeout,
        )
        status_code = int(response.status_code)
        if 400 <= status_code < 500 and status_code != 429:
            context.log_event("tool_call", "tavily_fast_fail", {
                              "status_code": status_code, "degraded": True})
            return []
        if status_code == 429 or status_code >= 500:
            raise RetryableStatusCodeError(
                status_code, f"Tavily retryable status: {status_code}")

        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
        clues: list[dict[str, str]] = []
        for item in results[:5]:
            clues.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "content": str(item.get("content") or "")[:500],
                }
            )
        return clues

    try:
        clues = _request_once()
        context.log_event("tool_call", "tavily_finish",
                          {"result_count": len(clues)})
        return clues
    except Exception as exc:
        context.log_event(
            "tool_call",
            "tavily_error",
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "status_code": extract_status_code(exc),
                "degraded": True,
            },
        )
        return []


def format_web_context(clues: list[dict[str, str]]) -> str:
    if not clues:
        return "(no external web clues available)"
    lines: list[str] = []
    for idx, clue in enumerate(clues, 1):
        topic = str(clue.get("topic") or "general").strip()
        query = str(clue.get("query") or "").strip()
        lines.append(f"Topic: {topic}")
        if query:
            lines.append(f"Query: {query}")
        lines.append(f"[{idx}] {clue.get('title', '').strip()}")
        lines.append(f"URL: {clue.get('url', '').strip()}")
        lines.append(f"Summary: {clue.get('content', '').strip()}")
        lines.append("")
    return "\n".join(lines).strip()


def research_stage(context: PipelineContext, bypass_autoresearch: bool = False) -> ResearchResult:
    web_clues = _search_web_clues(context, context.request.scene_prompt)
    if bypass_autoresearch:
        context.log_event(
            "research",
            "bypass_autoresearch",
            {
                "enabled": True,
                "reason": "Skip external LLM research and use deterministic local candidates.",
            },
        )
        result = _build_local_research_result(
            context.request.scene_prompt, web_clues)
        context.write_json("research.json", result.to_dict())
        context.write_json("research_web_clues.json", {"clues": web_clues})
        return result

    research_prompt = RESEARCH_PROMPT_TEMPLATE.format(
        scene_prompt=context.request.scene_prompt,
        web_context=format_web_context(web_clues),
    )
    payload = call_openrouter_json(context, research_prompt)
    payload_candidates = payload.get("candidates")
    if not isinstance(payload_candidates, list) or not payload_candidates:
        detail = {
            "error_code": "ResearchCandidatesInvalid",
            "message": "Research response has no valid candidates.",
            "payload_keys": sorted(payload.keys()),
            "candidate_count": len(payload_candidates) if isinstance(payload_candidates, list) else None,
            "stage": "research",
        }
        context.log_event("research", "invalid_candidates", detail)
        raise BusinessLogicError(json.dumps(detail, ensure_ascii=False))

    candidates: list[CandidateMethod] = []
    for item in payload_candidates:
        if not isinstance(item, dict):
            continue
        candidates.append(
            CandidateMethod(
                name=str(item.get("name") or "unknown_method"),
                description=str(item.get("description") or ""),
                parameters=item.get("parameters") if isinstance(
                    item.get("parameters"), dict) else {},
                rationale=str(item.get("rationale") or ""),
                sources=[str(source) for source in (
                    item.get("sources") or []) if str(source).strip()],
                confidence=float(item.get("confidence") or 0.0),
            )
        )
    if not candidates:
        detail = {
            "error_code": "ResearchCandidatesEmpty",
            "message": "Research response candidates are empty after parsing.",
            "candidate_count": 0,
            "stage": "research",
        }
        context.log_event("research", "empty_candidates_after_parse", detail)
        raise BusinessLogicError(json.dumps(detail, ensure_ascii=False))

    chosen_strategy = str(payload.get("chosen_strategy") or candidates[0].name)
    evaluation_metrics = [str(metric).strip() for metric in (
        payload.get("evaluation_metrics") or []) if str(metric).strip()]
    if not evaluation_metrics:
        evaluation_metrics = ["psnr", "ssim", "latency"]

    result = ResearchResult(
        scene_prompt=context.request.scene_prompt,
        candidates=candidates,
        chosen_strategy=chosen_strategy,
        evaluation_metrics=evaluation_metrics,
        evaluation_plan=str(payload.get(
            "evaluation_plan") or "Compute objective metrics on input/output pair and aggregate score."),
        summary=str(payload.get("summary") or f"Selected {chosen_strategy}"),
        sources=[str(source) for source in (payload.get("sources") or []) if str(source).strip()] or [
            clue.get("url", "") for clue in web_clues if clue.get("url")
        ],
    )
    context.write_json("research.json", result.to_dict())
    context.write_json("research_web_clues.json", {"clues": web_clues})
    return result
