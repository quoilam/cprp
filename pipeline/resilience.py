from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Protocol, TypeVar

import requests
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_exponential, wait_random

try:
    import openai
except Exception:  # pragma: no cover - fallback for environments without openai
    openai = None


T = TypeVar("T")


class EventLogger(Protocol):
    def log_event(self, category: str, event: str, payload: dict[str, Any] | None = None) -> Any:
        ...


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    max_retries: int
    max_attempts: int
    initial_delay: float
    max_delay: float
    jitter: float


class RetryableStatusCodeError(RuntimeError):
    def __init__(self, status_code: int, message: str | None = None):
        self.status_code = int(status_code)
        super().__init__(message or f"retryable http status: {status_code}")


class ResponseTruncatedError(ValueError):
    pass


class ModelResponseParseError(ValueError):
    def __init__(self, message: str, *, retryable: bool = True):
        self.retryable = retryable
        super().__init__(message)


class ParameterValidationError(ValueError):
    pass


class ContractValidationError(ValueError):
    pass


class BusinessLogicError(RuntimeError):
    pass


def build_retry_policy(
    *,
    max_retries: int | None = None,
    max_attempts: int | None = None,
    initial_delay: float,
    max_delay: float,
    jitter: float,
) -> RetryPolicy:
    if max_retries is None and max_attempts is None:
        raise ParameterValidationError("either max_retries or max_attempts must be provided")

    retries: int
    attempts: int

    if max_retries is not None:
        retries = max(0, int(max_retries))
        attempts = retries + 1
    else:
        attempts = max(1, int(max_attempts or 1))
        retries = attempts - 1

    if max_retries is not None and max_attempts is not None:
        normalized_attempts = max(1, int(max_attempts))
        if normalized_attempts != attempts:
            raise ParameterValidationError("max_retries and max_attempts are inconsistent")

    initial = max(0.0, float(initial_delay))
    max_wait = max(initial, float(max_delay))
    random_jitter = max(0.0, float(jitter))
    return RetryPolicy(
        max_retries=retries,
        max_attempts=attempts,
        initial_delay=initial,
        max_delay=max_wait,
        jitter=random_jitter,
    )


def build_retry_policy_from_config(config: Any, *, kind: str) -> RetryPolicy:
    if kind not in {"llm", "http"}:
        raise ValueError("kind must be either 'llm' or 'http'")

    max_retries = getattr(config, "llm_max_retries", 3) if kind == "llm" else getattr(config, "http_max_retries", 3)
    return build_retry_policy(
        max_retries=max_retries,
        initial_delay=getattr(config, "retry_initial_delay", 1.0),
        max_delay=getattr(config, "retry_max_delay", 8.0),
        jitter=getattr(config, "retry_jitter", 0.2),
    )


def build_wait_strategy(initial_delay: float, max_delay: float, jitter: float):
    wait_strategy = wait_exponential(
        multiplier=max(initial_delay, 0.0),
        min=max(initial_delay, 0.0),
        max=max(max_delay, initial_delay, 0.0),
    )
    if jitter > 0:
        return wait_strategy + wait_random(0, jitter)
    return wait_strategy


def _is_openai_exc(exc: BaseException, class_name: str) -> bool:
    if openai is None:
        return False
    exc_type = getattr(openai, class_name, None)
    return isinstance(exc, exc_type) if isinstance(exc_type, type) else False


def extract_status_code(exc: BaseException) -> int | None:
    if isinstance(exc, RetryableStatusCodeError):
        return exc.status_code

    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code

    for attr in ("status_code", "http_status", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code

    return None


def is_retryable_http_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


def is_non_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, (ParameterValidationError, ContractValidationError, BusinessLogicError)):
        return True
    if _is_openai_exc(exc, "BadRequestError"):
        return True
    status_code = extract_status_code(exc)
    return status_code is not None and 400 <= status_code < 500 and status_code != 429


def is_retryable_exception(exc: BaseException) -> bool:
    if is_non_retryable_exception(exc):
        return False

    if isinstance(exc, (ResponseTruncatedError, RetryableStatusCodeError)):
        return True

    if isinstance(exc, ModelResponseParseError):
        return exc.retryable

    status_code = extract_status_code(exc)
    if is_retryable_http_status(status_code):
        return True

    if _is_openai_exc(exc, "RateLimitError"):
        return True
    if _is_openai_exc(exc, "APITimeoutError"):
        return True
    if _is_openai_exc(exc, "APIConnectionError"):
        return True
    if _is_openai_exc(exc, "APIStatusError"):
        return is_retryable_http_status(status_code)

    return isinstance(exc, (ConnectionError, TimeoutError, requests.ConnectionError, requests.Timeout))


def classify_error_type(exc: BaseException) -> str:
    status_code = extract_status_code(exc)
    if status_code is not None:
        return f"http_{status_code}"
    return type(exc).__name__


def build_retry_summary(
    *,
    stage: str,
    operation: str,
    attempt: int,
    max_retries: int,
    max_attempts: int,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "operation": operation,
        "attempt": attempt,
        "max_retries": max_retries,
        "max_attempts": max_attempts,
        "retryable": is_retryable_exception(error),
        "error_type": classify_error_type(error),
        "error_code": type(error).__name__,
        "error_message": str(error)[:400],
    }


def _build_before_sleep_callback(
    *,
    context: EventLogger | None,
    stage: str,
    operation: str,
    max_retries: int,
    max_attempts: int,
) -> Callable[[RetryCallState], None] | None:
    if context is None:
        return None

    def _before_sleep(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            return
        exception = retry_state.outcome.exception()
        if exception is None:
            return

        backoff_seconds = 0.0
        if retry_state.next_action is not None and retry_state.next_action.sleep is not None:
            backoff_seconds = float(retry_state.next_action.sleep)

        context.log_event(
            "retry",
            "attempt",
            {
                "stage": stage,
                "operation": operation,
                "attempt": retry_state.attempt_number,
                "max_retries": max_retries,
                "max_attempts": max_attempts,
                "backoff_seconds": backoff_seconds,
                "retryable": is_retryable_exception(exception),
                "error_type": classify_error_type(exception),
            },
        )

    return _before_sleep


def build_retry_decorator(
    *,
    context: EventLogger | None,
    stage: str,
    operation: str,
    policy: RetryPolicy,
    retry_predicate: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    predicate = retry_predicate or is_retryable_exception
    before_sleep_callback = _build_before_sleep_callback(
        context=context,
        stage=stage,
        operation=operation,
        max_retries=policy.max_retries,
        max_attempts=policy.max_attempts,
    )

    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        tenacity_wrapped = retry(
            reraise=True,
            stop=stop_after_attempt(policy.max_attempts),
            wait=build_wait_strategy(policy.initial_delay, policy.max_delay, policy.jitter),
            retry=retry_if_exception(predicate),
            before_sleep=before_sleep_callback,
        )(func)

        @wraps(func)
        def _wrapped(*args: Any, **kwargs: Any) -> T:
            try:
                return tenacity_wrapped(*args, **kwargs)
            except Exception as exc:
                if context is not None:
                    retry_info = getattr(getattr(tenacity_wrapped, "retry", None), "statistics", {})
                    attempt = int(retry_info.get("attempt_number", 1))
                    context.log_event(
                        "retry",
                        "terminal_failure",
                        build_retry_summary(
                            stage=stage,
                            operation=operation,
                            attempt=attempt,
                            max_retries=policy.max_retries,
                            max_attempts=policy.max_attempts,
                            error=exc,
                        ),
                    )
                raise

        return _wrapped

    return _decorator
