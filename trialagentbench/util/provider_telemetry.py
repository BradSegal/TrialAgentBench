"""Append-only provider request telemetry and strict aggregation."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import ValidationError

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1
from trialagentbench.contracts.core.runs import (
    PROVIDER_ERROR_DETAIL_CHARACTERS,
    ProviderLogRecoveryV1,
    ProviderRequestEventV1,
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
)
from trialagentbench.io.exclusive_publication import append_jsonl, append_jsonl_model
from trialagentbench.io.json import loads_json, write_json_model
from trialagentbench.ports import LLMProvider, LLMResponse, RetryTelemetry
from trialagentbench.ports.llm_provider import JsonObject, ProviderRequestError, ProviderUnavailableError
from trialagentbench.util.provider_redaction import (
    redact_provider_exception,
    redact_provider_text,
    redact_provider_value,
)
from trialagentbench.util.reported_cost import ReportedCostThresholdReached, ReportedCostUnavailable
from trialagentbench.util.runtime_context import RuntimeDeadline

ProviderFailureType = Literal["timeout", "provider_error", "cancelled"]
ProviderContextManagement = Literal["unreported", "provider_managed", "local_complete", "local_archived"]


@dataclass(frozen=True)
class ProviderRequestHandle:
    """Identity shared by the start and terminal events for one request."""

    path: Path
    request_id: str
    benchmark: Literal["trialeval", "trialdev"]
    unit_id: str
    phase_id: str
    step_id: str
    turn_index: int
    requested_model: str
    provider_route: str
    context_management: ProviderContextManagement
    source_context_messages: int
    transmitted_context_messages: int
    source_context_characters: int
    transmitted_context_characters: int
    requested_completion_tokens: int
    provider_capability: ProviderModelCapabilityV1 | None = None


@dataclass(frozen=True)
class ProviderTurnResult:
    """One provider response and its observed wall-clock duration."""

    response: LLMResponse
    elapsed_seconds: float


def generate_provider_turn_v1(
    *,
    provider: LLMProvider,
    messages: Sequence[JsonObject],
    tools: Sequence[JsonObject] | None,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float | None,
    tool_choice: Literal["auto", "required"],
    log_path: Path | None,
    benchmark: Literal["trialeval", "trialdev"],
    unit_id: str,
    phase_id: str,
    step_id: str,
    turn_index: int,
    context_management: ProviderContextManagement | None = None,
    source_context_messages: int | None = None,
    source_context_characters: int | None = None,
    provider_capability: ProviderModelCapabilityV1 | None = None,
    deadline: RuntimeDeadline | None = None,
) -> ProviderTurnResult:
    """Execute and, when requested, record one provider request lifecycle."""

    with deadline.harness_work() if deadline is not None else nullcontext():
        transmitted_context_messages = len(messages)
        transmitted_context_characters = sum(
            len(json.dumps(message, ensure_ascii=True, sort_keys=True, default=str)) for message in messages
        )
        observed_source_messages = (
            transmitted_context_messages if source_context_messages is None else source_context_messages
        )
        observed_source_characters = (
            transmitted_context_characters if source_context_characters is None else source_context_characters
        )
        observed_management: ProviderContextManagement = context_management or "local_complete"
        handle = (
            None
            if log_path is None
            else start_provider_request_v1(
                path=log_path,
                benchmark=benchmark,
                unit_id=unit_id,
                phase_id=phase_id,
                step_id=step_id,
                turn_index=turn_index,
                requested_model=provider.model,
                provider_route=provider.telemetry_route,
                context_management=observed_management,
                source_context_messages=observed_source_messages,
                transmitted_context_messages=transmitted_context_messages,
                source_context_characters=observed_source_characters,
                transmitted_context_characters=transmitted_context_characters,
                requested_completion_tokens=max_tokens,
                provider_capability=provider_capability,
            )
        )
    started = time.monotonic()
    response: LLMResponse | None = None
    try:
        if provider_capability is not None:
            if provider_capability.model_id != provider.model:
                raise ValueError("Provider capability does not match the requested model.")
            capacity = provider_capability.maximum_completion_tokens
            if capacity is not None and max_tokens > capacity:
                raise ValueError("Requested completion exceeds the source-bound provider capacity.")
        request_timeout = timeout_seconds
        if deadline is not None:
            remaining = deadline.remaining()
            if remaining is not None:
                request_timeout = remaining if request_timeout is None else min(request_timeout, remaining)
        response = provider.generate_turn(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=request_timeout,
            tool_choice=tool_choice,
        )
    finally:
        error = sys.exception()
        if error is not None:
            redact_provider_exception(error)
        elapsed = time.monotonic() - started
        invalid_deferral: ProviderRequestError | None = None
        if isinstance(error, ProviderUnavailableError):
            if (
                error.response_received
                or isinstance(getattr(error, "raw_response", None), Mapping)
                or (deadline is not None and deadline.provider_unavailability_policy != error.policy)
                or error.unanswered_transport_seconds is None
                or not math.isfinite(error.unanswered_transport_seconds)
                or not 0 <= error.unanswered_transport_seconds <= elapsed
            ):
                # Fail closed, but still close the append-only request lifecycle
                # and preserve any received payload. Never refund or expose this
                # inconsistent error as an automatically recoverable deferral.
                invalid_deferral = ProviderRequestError(
                    "Provider deferral evidence or runtime policy is inconsistent; operator review required.",
                    retry_telemetry=error.retry_telemetry,
                )
                invalid_deferral.response_received = error.response_received
                invalid_deferral.raw_response = getattr(error, "raw_response", None)
                invalid_deferral.__cause__ = error
                error = invalid_deferral
            else:
                error.elapsed_seconds = elapsed
                if deadline is not None:
                    deadline.exclude_unanswered_transport(error.unanswered_transport_seconds)
                    error.scientific_clock_excluded_seconds = error.unanswered_transport_seconds
        with deadline.harness_work() if deadline is not None else nullcontext():
            if handle is not None:
                if response is None:
                    fail_provider_request_v1(
                        handle,
                        elapsed_seconds=elapsed,
                        failure_type=provider_failure_type_v1(error),
                        error=error,
                    )
                else:
                    try:
                        succeed_provider_request_v1(
                            handle,
                            elapsed_seconds=elapsed,
                            response=response,
                        )
                    except (TypeError, ValueError) as error:
                        redact_provider_exception(error)
                        fail_provider_request_v1(
                            handle,
                            elapsed_seconds=elapsed,
                            failure_type="provider_error",
                            error=error,
                        )
                        raise
        if invalid_deferral is not None:
            raise invalid_deferral
    if response is None:
        raise RuntimeError("Provider request returned without a response or exception.")
    return ProviderTurnResult(response=response, elapsed_seconds=elapsed)


def _continuation_path(path: Path) -> Path:
    """Append in a new segment if death left the original without a final newline."""

    while path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            if stream.read(1) == b"\n":
                break
        path = path.with_name(f"{path.stem}.continued.jsonl")
    return path


def start_provider_request_v1(
    *,
    path: Path,
    benchmark: Literal["trialeval", "trialdev"],
    unit_id: str,
    phase_id: str,
    step_id: str,
    turn_index: int,
    requested_model: str,
    provider_route: str,
    context_management: ProviderContextManagement = "unreported",
    source_context_messages: int = 0,
    transmitted_context_messages: int = 0,
    source_context_characters: int = 0,
    transmitted_context_characters: int = 0,
    requested_completion_tokens: int = 0,
    provider_capability: ProviderModelCapabilityV1 | None = None,
) -> ProviderRequestHandle:
    """Persist request identity and route before provider transport starts."""

    # A cancelled request can be retried at the same checkpoint and turn.
    # Transport attempts need distinct identities even when their task identity is unchanged.
    request_id = f"{Path(path).name}:{benchmark}:{unit_id}:{phase_id}:{step_id}:{turn_index}:{uuid4().hex}"
    append_path = _continuation_path(Path(path))
    handle = ProviderRequestHandle(
        path=append_path,
        request_id=request_id,
        benchmark=benchmark,
        unit_id=unit_id,
        phase_id=phase_id,
        step_id=step_id,
        turn_index=turn_index,
        requested_model=requested_model,
        provider_route=provider_route,
        context_management=context_management,
        source_context_messages=source_context_messages,
        transmitted_context_messages=transmitted_context_messages,
        source_context_characters=source_context_characters,
        transmitted_context_characters=transmitted_context_characters,
        requested_completion_tokens=requested_completion_tokens,
        provider_capability=provider_capability,
    )
    append_jsonl_model(
        handle.path,
        _request_event(
            handle,
            status="started",
            elapsed_seconds=None,
            usage_status="not_applicable",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_attempts=0,
            transient_failure_count=0,
            backoff_seconds=0.0,
        ),
    )
    return handle


def succeed_provider_request_v1(
    handle: ProviderRequestHandle,
    *,
    elapsed_seconds: float,
    response: LLMResponse,
) -> ProviderRequestEventV1:
    """Retain the response before validating accounting; bad usage must not discard an answer."""

    _append_provider_payload(
        handle,
        response.raw_response,
        elapsed_seconds=elapsed_seconds,
        attempt_failures=response.metadata.attempt_failures,
        usage=response.usage,
    )
    usage = response.usage
    try:
        event = _request_event(
            handle,
            status="succeeded",
            elapsed_seconds=elapsed_seconds,
            usage_status="reported" if usage is not None else "not_reported",
            prompt_tokens=_required_usage_integer(usage, "prompt_tokens") if usage is not None else 0,
            completion_tokens=_required_usage_integer(usage, "completion_tokens") if usage is not None else 0,
            total_tokens=_required_usage_integer(usage, "total_tokens") if usage is not None else 0,
            response=response,
            request_attempts=response.metadata.request_attempts,
            transient_failure_count=response.metadata.transient_failure_count,
            backoff_seconds=response.metadata.backoff_seconds,
        )
    except (TypeError, ValueError) as error:
        if usage is None:
            raise
        # Revalidate everything else unchanged: unrelated metadata failures must still
        # propagate. Never mutate the response delivered to the agent or invent counts.
        event = _request_event(
            handle,
            status="succeeded",
            elapsed_seconds=elapsed_seconds,
            usage_status="invalid",
            usage_error=_error_detail(error),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response=replace(response, usage=None),
            request_attempts=response.metadata.request_attempts,
            transient_failure_count=response.metadata.transient_failure_count,
            backoff_seconds=response.metadata.backoff_seconds,
        )
    append_jsonl_model(handle.path, event)
    return event


def fail_provider_request_v1(
    handle: ProviderRequestHandle,
    *,
    elapsed_seconds: float,
    failure_type: ProviderFailureType,
    error: BaseException | None = None,
) -> ProviderRequestEventV1:
    """Append a typed failed terminal event for a started request."""

    retry = retry_telemetry_from_error_v1(error)
    raw_response = getattr(error, "raw_response", None)
    _append_provider_payload(
        handle,
        raw_response if isinstance(raw_response, Mapping) else None,
        elapsed_seconds=elapsed_seconds,
        attempt_failures=retry.attempt_failures,
    )
    event = _request_event(
        handle,
        status="failed",
        elapsed_seconds=elapsed_seconds,
        usage_status="not_applicable",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request_attempts=retry.request_attempts,
        transient_failure_count=retry.transient_failure_count,
        backoff_seconds=retry.backoff_seconds,
        failure_type=failure_type,
        error=error,
    )
    append_jsonl_model(handle.path, event)
    return event


def _append_provider_payload(
    handle: ProviderRequestHandle,
    payload: JsonObject | None,
    *,
    elapsed_seconds: float,
    attempt_failures: tuple[JsonObject, ...] = (),
    usage: JsonObject | None = None,
) -> None:
    """Retain returned JSON and failed attempts for diagnosis, apart from scored telemetry and request headers."""
    if payload is None and not attempt_failures and usage is None:
        return
    stem = handle.path.stem.replace("provider_responses", "provider_payloads")
    append_jsonl(
        _continuation_path(handle.path.with_name(f"{stem}.raw.jsonl")),
        {
            "request_id": handle.request_id,
            "unit_id": handle.unit_id,
            "turn_index": handle.turn_index,
            "elapsed_seconds": elapsed_seconds,
            "response": redact_provider_value(dict(payload)) if payload is not None else None,
            "usage": redact_provider_value(dict(usage)) if usage is not None else None,
            "attempt_failures": redact_provider_value([dict(item) for item in attempt_failures]),
        },
    )


def provider_failure_type_v1(error: BaseException | None) -> ProviderFailureType:
    """Map a propagated provider-call failure to the public failure taxonomy."""

    if isinstance(error, TimeoutError):
        return "timeout"
    # RunStopRequested and CampaignCapacityStop inherit KeyboardInterrupt.
    # Reported-cost guards also stop locally before contacting the provider.
    if isinstance(error, (KeyboardInterrupt, SystemExit, ReportedCostThresholdReached, ReportedCostUnavailable)):
        return "cancelled"
    return "provider_error"


def retry_telemetry_from_error_v1(error: BaseException | None) -> RetryTelemetry:
    """Return exact retry observations attached by a live provider adapter."""

    telemetry = getattr(error, "retry_telemetry", None)
    if telemetry is None:
        return RetryTelemetry(
            request_attempts=1,
            transient_failure_count=0,
            backoff_seconds=0.0,
        )
    if not isinstance(telemetry, RetryTelemetry):
        raise TypeError("Provider exception retry_telemetry must use the typed RetryTelemetry contract.")
    return telemetry


def summarize_provider_telemetry_v1(
    *,
    run_root: Path,
    coverage: RunCoverageV1,
) -> ProviderTelemetrySummaryV1:
    """Validate request joins and aggregate terminal provider events."""

    root = Path(run_root)
    paths, archived_paths = provider_log_paths_v1(root)
    recoveries: list[ProviderLogRecoveryV1] = []

    def read_logs(source_paths: tuple[Path, ...]) -> tuple[list[ProviderRequestEventV1], int]:
        terminal: list[ProviderRequestEventV1] = []
        all_events: list[ProviderRequestEventV1] = []
        interrupted_count = 0
        for path in source_paths:
            events, tail = _read_provider_events(path)
            all_events.extend(events)
            completed = (
                [event for event in events if event.status != "started"]
                if source_paths == paths
                else _validated_terminal_events(events)
            )
            completed_ids = {event.request_id for event in completed}
            interrupted = tuple(event.request_id for event in events if event.request_id not in completed_ids)
            interrupted_count += len(interrupted)
            terminal.extend(completed)
            if interrupted or tail:
                recoveries.append(
                    ProviderLogRecoveryV1(
                        source_file=path.relative_to(root).as_posix(),
                        interrupted_request_ids=interrupted,
                        torn_tail_bytes=len(tail),
                        torn_tail_sha256=hashlib.sha256(tail).hexdigest() if tail else None,
                    )
                )
        if source_paths == paths:
            _validated_terminal_events(all_events)
        return terminal, interrupted_count

    terminal_events, interrupted_count = read_logs(paths)
    archived_terminal_events, archived_interrupted_count = read_logs(archived_paths)
    succeeded = [event for event in terminal_events if event.status == "succeeded"]
    failed = [event for event in terminal_events if event.status == "failed"]
    archived_failed = [event for event in archived_terminal_events if event.status == "failed"]
    summary = ProviderTelemetrySummaryV1(
        run_identity_sha256=coverage.run_identity_sha256,
        schedule_sha256=coverage.schedule_sha256,
        unit_ids=coverage.unit_ids,
        completed_unit_ids=coverage.completed_unit_ids,
        response_count=len(succeeded),
        interrupted_request_count=interrupted_count,
        archived_interrupted_request_count=archived_interrupted_count,
        log_recoveries=tuple(recoveries),
        locally_archived_request_count=sum(event.context_management == "local_archived" for event in terminal_events),
        completion_limit_reached_count=sum(event.completion_limit_reached is True for event in succeeded),
        source_context_characters=sum(event.source_context_characters for event in terminal_events),
        transmitted_context_characters=sum(event.transmitted_context_characters for event in terminal_events),
        requested_completion_tokens=sum(event.requested_completion_tokens for event in terminal_events),
        failed_request_count=len(failed),
        failure_type_counts=_failure_type_counts(failed),
        responses_with_usage=sum(event.usage_status == "reported" for event in succeeded),
        responses_with_invalid_usage=sum(event.usage_status == "invalid" for event in succeeded),
        responses_with_response_id=sum(event.response_id is not None for event in succeeded),
        responses_with_reported_cost=sum(event.reported_cost_usd is not None for event in succeeded),
        prompt_tokens=sum(event.prompt_tokens for event in succeeded),
        completion_tokens=sum(event.completion_tokens for event in succeeded),
        total_tokens=sum(event.total_tokens for event in succeeded),
        compaction_requests=sum(event.compaction_requests for event in succeeded),
        compaction_input_tokens=sum(event.compaction_input_tokens for event in succeeded),
        compaction_output_tokens=sum(event.compaction_output_tokens for event in succeeded),
        reported_cost_usd=sum(event.reported_cost_usd or 0.0 for event in succeeded),
        elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in terminal_events),
        request_attempt_count=sum(event.request_attempts for event in terminal_events),
        transient_failure_count=sum(event.transient_failure_count for event in terminal_events),
        backoff_seconds=sum(event.backoff_seconds for event in terminal_events),
        requested_models=sorted({event.requested_model for event in terminal_events}),
        returned_models=sorted({event.returned_model for event in succeeded if event.returned_model is not None}),
        upstream_providers=sorted(
            {event.upstream_provider for event in succeeded if event.upstream_provider is not None}
        ),
        source_files=[path.relative_to(root).as_posix() for path in paths],
        archived_request_count=len(archived_terminal_events),
        archived_failed_request_count=len(archived_failed),
        archived_failure_type_counts=_failure_type_counts(archived_failed),
        archived_request_attempt_count=sum(event.request_attempts for event in archived_terminal_events),
        archived_transient_failure_count=sum(event.transient_failure_count for event in archived_terminal_events),
        archived_backoff_seconds=sum(event.backoff_seconds for event in archived_terminal_events),
        archived_elapsed_seconds=sum(event.elapsed_seconds or 0.0 for event in archived_terminal_events),
        archived_source_files=[path.relative_to(root).as_posix() for path in archived_paths],
    )
    write_json_model(root / "provider_telemetry_summary.json", summary)
    return summary


def read_provider_request_log_v1(path: Path) -> tuple[tuple[ProviderRequestEventV1, ...], int]:
    """Return validated complete events and the byte count of an untouched torn tail."""

    events, tail = _read_provider_events(Path(path))
    _validated_terminal_events(events)
    return tuple(events), len(tail)


def read_provider_terminal_events_v1(path: Path) -> tuple[ProviderRequestEventV1, ...]:
    """Read and validate the terminal provider events from one lifecycle log."""

    return tuple(_validated_terminal_events(_read_provider_events(Path(path))[0]))


def _is_authoritative_provider_log(path: Path, root: Path) -> bool:
    """Exclude retained failed-attempt archives from authoritative run totals."""

    relative_parts = path.relative_to(root).parts[:-1]
    return not any(
        part.lower().startswith(("archive", "failed_attempt", "failed-attempt"))
        or part.lower() in {"attempts", "superseded"}
        for part in relative_parts
    )


def provider_log_paths_v1(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Partition run logs into authoritative execution and retained archives."""

    resolved = Path(root)
    all_paths = tuple(sorted(resolved.rglob("*provider_responses*.jsonl"), key=lambda path: path.as_posix()))
    authoritative = tuple(path for path in all_paths if _is_authoritative_provider_log(path, resolved))
    archived = tuple(path for path in all_paths if path not in authoritative)
    return authoritative, archived


def _request_event(
    handle: ProviderRequestHandle,
    *,
    status: Literal["started", "succeeded", "failed"],
    elapsed_seconds: float | None,
    usage_status: Literal["not_applicable", "reported", "not_reported", "invalid"],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    request_attempts: int,
    transient_failure_count: int,
    backoff_seconds: float,
    response: LLMResponse | None = None,
    usage_error: str | None = None,
    failure_type: ProviderFailureType | None = None,
    error: BaseException | None = None,
) -> ProviderRequestEventV1:
    metadata = response.metadata if response is not None else None
    return ProviderRequestEventV1(
        request_id=handle.request_id,
        status=status,
        benchmark=handle.benchmark,
        unit_id=handle.unit_id,
        phase_id=handle.phase_id,
        step_id=handle.step_id,
        turn_index=handle.turn_index,
        elapsed_seconds=elapsed_seconds,
        requested_model=handle.requested_model,
        provider_route=handle.provider_route,
        context_management=handle.context_management,
        source_context_messages=handle.source_context_messages,
        transmitted_context_messages=handle.transmitted_context_messages,
        source_context_characters=handle.source_context_characters,
        transmitted_context_characters=handle.transmitted_context_characters,
        requested_completion_tokens=handle.requested_completion_tokens,
        provider_capability=handle.provider_capability,
        response_received=response is not None
        or getattr(error, "response_received", False) is True
        or isinstance(getattr(error, "raw_response", None), Mapping),
        response_id=metadata.response_id if metadata is not None else None,
        returned_model=metadata.returned_model if metadata is not None else None,
        upstream_provider=metadata.upstream_provider if metadata is not None else None,
        finish_reason=metadata.finish_reason if metadata is not None else None,
        completion_limit_reached=_completion_limit_reached(metadata.finish_reason if metadata is not None else None),
        created_unix=metadata.created_unix if metadata is not None else None,
        usage_status=usage_status,
        usage_error=usage_error,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        compaction_requests=_optional_usage_integer(response, "compaction_requests"),
        compaction_input_tokens=_optional_usage_integer(response, "compaction_input_tokens"),
        compaction_output_tokens=_optional_usage_integer(response, "compaction_output_tokens"),
        cached_tokens=_optional_usage_integer(response, "cached_tokens"),
        reasoning_tokens=_optional_usage_integer(response, "reasoning_tokens"),
        reasoning=(
            redact_provider_text(response.reasoning)
            if response is not None and response.reasoning is not None
            else None
        ),
        model_outcome=response.model_outcome if response is not None else None,
        reported_cost_usd=metadata.reported_cost_usd if metadata is not None else None,
        request_attempts=request_attempts,
        transient_failure_count=transient_failure_count,
        backoff_seconds=backoff_seconds,
        failure_type=failure_type,
        exception_type=type(error).__name__ if error is not None else None,
        http_status_code=_http_status_code(error),
        error_detail=_error_detail(error),
        unavailability_policy=error.policy if isinstance(error, ProviderUnavailableError) else None,
        scientific_clock_excluded_seconds=(
            error.scientific_clock_excluded_seconds if isinstance(error, ProviderUnavailableError) else 0.0
        ),
        retry_after_seconds=error.retry_after_seconds if isinstance(error, ProviderUnavailableError) else None,
    )


def _error_detail(error: BaseException | None) -> str | None:
    """Return the head of each chained provider error message and response body."""
    parts: list[str] = []
    while error is not None:
        parts.append(f"{type(error).__name__}: {error}")
        body = getattr(error, "body", None)
        if body is not None:
            parts.append(f"body: {json.dumps(body, default=str, sort_keys=True)}")
        error = error.__cause__
    return redact_provider_text(" | ".join(parts))[:PROVIDER_ERROR_DETAIL_CHARACTERS] or None


def _completion_limit_reached(finish_reason: str | None) -> bool | None:
    """Interpret only provider finish reasons with stable completion semantics."""

    if finish_reason is None:
        return None
    terminal_reason = finish_reason.casefold().rsplit(":", maxsplit=1)[-1]
    return terminal_reason in {"length", "max_tokens", "max_output_tokens"}


def _http_status_code(error: BaseException | None) -> int | None:
    """Return a bounded integer HTTP status exposed by a provider exception."""

    if error is None:
        return None
    value = getattr(error, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _failure_type_counts(
    events: list[ProviderRequestEventV1],
) -> dict[ProviderFailureType, int]:
    """Return complete typed counts for failed terminal events."""

    counts: dict[ProviderFailureType, int] = {
        "timeout": 0,
        "provider_error": 0,
        "cancelled": 0,
    }
    for event in events:
        if event.status != "failed" or event.failure_type is None:
            raise ValueError("Failure aggregation received a non-failed provider event.")
        counts[event.failure_type] += 1
    return counts


def _read_provider_events(path: Path) -> tuple[list[ProviderRequestEventV1], bytes]:
    """Read complete records; retain and identify only an unterminated torn tail.

    A malformed newline-terminated record or an invalid contract still fails
    loudly. Recovery never edits the log or manufactures a terminal event.
    """

    events: list[ProviderRequestEventV1] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = loads_json(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if not line.endswith(b"\n"):
                    return events, line
                raise ValueError(f"Invalid provider telemetry at {path}:{line_number}") from error
            try:
                events.append(ProviderRequestEventV1.model_validate(payload))
            except ValidationError as error:
                raise ValueError(f"Invalid provider telemetry at {path}:{line_number}") from error
    return events, b""


def _required_usage_integer(usage: object, key: str) -> int:
    if not isinstance(usage, Mapping):
        raise TypeError("Provider usage must be a mapping.")
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Provider usage {key} must be a non-negative integer, observed {value!r}.")
    return value


def _optional_usage_integer(response: LLMResponse | None, key: str) -> int:
    """Return an optional non-negative usage field from a successful response."""

    if response is None or response.usage is None:
        return 0
    value = response.usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Provider usage {key} must be a non-negative integer, observed {value!r}.")
    return value


def _validated_terminal_events(
    events: list[ProviderRequestEventV1],
) -> list[ProviderRequestEventV1]:
    starts: dict[str, ProviderRequestEventV1] = {}
    terminals: dict[str, ProviderRequestEventV1] = {}
    positions: dict[tuple[str, str], int] = {}
    for position, event in enumerate(events):
        target = starts if event.status == "started" else terminals
        if event.request_id in target:
            raise ValueError(f"Duplicate provider request {event.status} event: {event.request_id}.")
        target[event.request_id] = event
        positions[(event.request_id, "started" if event.status == "started" else "terminal")] = position
    if terminals.keys() - starts.keys():
        missing_terminal = sorted(starts.keys() - terminals.keys())
        missing_start = sorted(terminals.keys() - starts.keys())
        raise ValueError(
            "Provider request telemetry has missing joins: "
            f"missing_terminal={missing_terminal}, missing_start={missing_start}."
        )
    identity_fields = (
        "benchmark",
        "unit_id",
        "phase_id",
        "step_id",
        "turn_index",
        "requested_model",
        "provider_route",
        "context_management",
        "source_context_messages",
        "transmitted_context_messages",
        "source_context_characters",
        "transmitted_context_characters",
        "requested_completion_tokens",
        "provider_capability",
    )
    for request_id, start in starts.items():
        terminal = terminals.get(request_id)
        if terminal is None:
            continue
        if positions[(request_id, "started")] >= positions[(request_id, "terminal")]:
            raise ValueError(f"Provider request terminal precedes its start event: {request_id}.")
        if any(getattr(start, field) != getattr(terminal, field) for field in identity_fields):
            raise ValueError(f"Provider request identity or route mismatch for {request_id}.")
    return [event for event in events if event.status != "started"]


__all__ = [
    "ProviderRequestHandle",
    "ProviderTurnResult",
    "fail_provider_request_v1",
    "generate_provider_turn_v1",
    "provider_log_paths_v1",
    "provider_failure_type_v1",
    "read_provider_terminal_events_v1",
    "read_provider_request_log_v1",
    "retry_telemetry_from_error_v1",
    "start_provider_request_v1",
    "succeed_provider_request_v1",
    "summarize_provider_telemetry_v1",
]
