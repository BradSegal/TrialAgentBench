"""Denominator-complete summaries of benchmark calibration runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from math import fsum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.core.config import MemoryStrategyV1, ProcedureAssistanceV1, ToolChoiceV1
from trialagentbench.contracts.core.runs import (
    ProviderRequestEventV1,
    TrialEvalItemResultV1,
    TrialEvalRunConfigV1,
)
from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import canonical_payload_sha256, read_json, read_json_model
from trialagentbench.util.provider_telemetry import (
    provider_log_paths_v1,
    read_provider_terminal_events_v1,
)

BenchmarkV1 = Literal["trialeval", "trialdev"]
CalibrationContrastFactorV1 = Literal[
    "provider",
    "upstream_provider",
    "reasoning_effort",
    "decoding_seed",
    "procedure_assistance",
    "tool_choice",
    "memory_strategy",
    "turn_limit",
    "completion_token_limit",
    "context_character_limit",
    "send_temperature",
    "request_timeout_seconds",
    "watchdog_seconds",
    "provider_maximum_attempts",
    "provider_retry_base_delay_seconds",
    "reported_cost_stop_usd",
    "executor_image",
    "executor_limits_sha256",
    "worker_count",
]
ExecutionStatusV1 = Literal[
    "not_started",
    "incomplete_run",
    "success",
    "max_turns_reached",
    "failed",
    "completed",
    "model_noncompletion",
    "evaluator_error",
]
TokenAccountingStatusV1 = Literal["complete", "partial", "unknown"]


class CalibrationUnitV1(BaseModel):
    """One prospectively scheduled benchmark unit and its observed outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.calibration_unit/v2"] = "ariadne.calibration_unit/v2"
    benchmark: BenchmarkV1
    run_id: str = Field(min_length=1)
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_identity: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    decoding_seed: int | None = Field(default=None, ge=0)
    condition_id: str | None = Field(default=None, min_length=1)
    replicate_id: str | None = Field(default=None, min_length=1)
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only"
    tool_choice: ToolChoiceV1 = "auto"
    memory_strategy: MemoryStrategyV1 = "bounded"
    analysis_specification: str | None = Field(default=None, min_length=1)
    stream: str | None = Field(default=None, min_length=1)
    turn_limit: int = Field(ge=1)
    completion_token_limit: int = Field(ge=1)
    context_character_limit: int = Field(ge=1)
    send_temperature: bool
    request_timeout_seconds: float = Field(gt=0.0)
    watchdog_seconds: float = Field(gt=0.0)
    provider_maximum_attempts: int = Field(ge=1)
    provider_retry_base_delay_seconds: float = Field(ge=0.0)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)
    executor_image: str = Field(min_length=1)
    executor_limits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_count: int = Field(ge=1)
    execution_status: ExecutionStatusV1
    turns_used: int | None = Field(default=None, ge=0)
    primary_failure_origin: str | None = Field(default=None, min_length=1)
    provider_response_count: int = Field(ge=0)
    provider_failure_count: int = Field(ge=0)
    # Omitted fields remain unknown and are excluded from the checksum.
    provider_invalid_usage_count: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    provider_unreported_usage_count: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    token_accounting_status: TokenAccountingStatusV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    analysis_prompt_tokens: int = Field(ge=0)
    analysis_completion_tokens: int = Field(ge=0)
    analysis_reported_cost_usd: float = Field(ge=0.0)
    analysis_elapsed_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        """Require provider counts to retain all terminal requests."""

        if self.provider_failure_count > self.provider_response_count:
            raise ValueError("Provider failures cannot exceed terminal provider responses")
        missing = (self.provider_invalid_usage_count, self.provider_unreported_usage_count)
        if all(count is not None for count in missing):
            missing_total = sum(count for count in missing if count is not None)
            if missing_total > self.provider_response_count - self.provider_failure_count:
                raise ValueError("Invalid and unreported usage cannot exceed successful responses")
        if self.token_accounting_status == "complete" and (
            missing != (0, 0) or self.provider_failure_count or not self.provider_response_count
        ):
            raise ValueError("Complete token accounting requires observed responses with valid usage")
        return self


class CalibrationCellV1(BaseModel):
    """Exact request and independent-unit denominators for one cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkV1
    release_identity: str
    model: str
    provider: str
    upstream_provider: str | None = None
    reasoning_effort: str | None = None
    decoding_seed: int | None = None
    condition_ids: tuple[str, ...] = Field(min_length=1)
    procedure_assistance: ProcedureAssistanceV1
    tool_choice: ToolChoiceV1
    memory_strategy: MemoryStrategyV1
    analysis_specification: str | None = None
    stream: str | None = None
    turn_limit: int = Field(ge=1)
    completion_token_limit: int = Field(ge=1)
    context_character_limit: int = Field(ge=1)
    send_temperature: bool
    request_timeout_seconds: float = Field(gt=0.0)
    watchdog_seconds: float = Field(gt=0.0)
    provider_maximum_attempts: int = Field(ge=1)
    provider_retry_base_delay_seconds: float = Field(ge=0.0)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)
    executor_image: str = Field(min_length=1)
    executor_limits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_count: int = Field(ge=1)
    request_replicate_ids: tuple[str, ...] = Field(min_length=1)
    scheduled_requests: int = Field(ge=1)
    independent_unit_count: int = Field(ge=1)
    execution_completed: int = Field(ge=0)
    model_noncompletion: int = Field(ge=0)
    evaluator_error: int = Field(ge=0)
    provider_failures: int = Field(ge=0)
    provider_invalid_usage_count: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    provider_unreported_usage_count: int | None = Field(default=None, ge=0, exclude_if=lambda value: value is None)
    token_accounting_status: TokenAccountingStatusV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    analysis_tokens: int = Field(ge=0)
    observed_cost_usd: float = Field(ge=0.0)
    observed_elapsed_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_denominators(self) -> Self:
        """Require all cell counts and rates to use their declared denominators."""

        if self.request_replicate_ids != tuple(sorted(set(self.request_replicate_ids))):
            raise ValueError("Calibration request replicate identifiers must be sorted and unique")
        if self.execution_completed > self.scheduled_requests:
            raise ValueError("Calibration completions cannot exceed scheduled units")
        if self.independent_unit_count > self.scheduled_requests:
            raise ValueError("Independent units cannot outnumber scheduled requests")
        if self.token_accounting_status == "complete" and (
            self.provider_invalid_usage_count != 0
            or self.provider_unreported_usage_count != 0
            or self.provider_failures
        ):
            raise ValueError("Complete token accounting cannot include unknown, invalid or unreported usage")
        return self


class CalibrationSummaryV1(BaseModel):
    """Checksummed cross-run calibration table and exact cell summaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.calibration_summary/v2"] = "ariadne.calibration_summary/v2"
    units: tuple[CalibrationUnitV1, ...]
    cells: tuple[CalibrationCellV1, ...]
    run_count: int = Field(ge=1)
    scheduled_request_count: int = Field(ge=1)
    independent_unit_count: int = Field(ge=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        """Require unique units, exact totals, and a reproducible checksum."""

        identities = tuple((row.run_identity_sha256, row.unit_id) for row in self.units)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError("Calibration units must have unique sorted run and unit identities")
        if self.scheduled_request_count != len(self.units):
            raise ValueError("Calibration scheduled_request_count differs from its request rows")
        if sum(cell.scheduled_requests for cell in self.cells) != self.scheduled_request_count:
            raise ValueError("Calibration cells do not cover the scheduled request denominator")
        independent = {(row.release_identity, row.unit_id) for row in self.units}
        if self.independent_unit_count != len(independent):
            raise ValueError("Calibration independent_unit_count differs from its unit rows")
        if self.run_count != len({row.run_identity_sha256 for row in self.units}):
            raise ValueError("Calibration run_count differs from its unit rows")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("Calibration summary checksum mismatch")
        return self

    def with_checksum(self) -> CalibrationSummaryV1:
        """Return the summary with its canonical content checksum."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


class CalibrationConditionV1(BaseModel):
    """Expected execution factors for one prospectively named condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    decoding_seed: int | None = Field(default=None, ge=0)
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only"
    tool_choice: ToolChoiceV1 = "auto"
    memory_strategy: MemoryStrategyV1 = "bounded"
    turn_limit: int = Field(ge=1)
    completion_token_limit: int = Field(ge=1)
    context_character_limit: int = Field(ge=1)
    send_temperature: bool
    request_timeout_seconds: float = Field(gt=0.0)
    watchdog_seconds: float = Field(gt=0.0)
    provider_maximum_attempts: int = Field(ge=1)
    provider_retry_base_delay_seconds: float = Field(ge=0.0)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)
    executor_image: str = Field(min_length=1)
    executor_limits_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    worker_count: int = Field(ge=1)


class CalibrationPanelV1(BaseModel):
    """Prospective matched panel checked against completed run artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.calibration_panel/v1"] = "ariadne.calibration_panel/v1"
    panel_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    benchmark: BenchmarkV1
    release_identity: str = Field(min_length=1)
    model: str = Field(min_length=1)
    contrast_factor: CalibrationContrastFactorV1
    conditions: tuple[CalibrationConditionV1, ...] = Field(min_length=2)
    request_replicate_ids: tuple[str, ...] = Field(min_length=1)
    unit_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_panel(self) -> Self:
        """Require unique factors in the prospectively declared panel."""

        condition_ids = tuple(condition.condition_id for condition in self.conditions)
        for name, values in (
            ("condition_ids", condition_ids),
            ("request_replicate_ids", self.request_replicate_ids),
            ("unit_ids", self.unit_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Calibration panel {name} must be unique")
        factor_names = tuple(
            factor_name for factor_name in CalibrationConditionV1.model_fields if factor_name != "condition_id"
        )
        for factor_name in factor_names:
            values = tuple(getattr(condition, factor_name) for condition in self.conditions)
            if factor_name == self.contrast_factor:
                if len(set(values)) < 2:
                    raise ValueError("Calibration panel contrast factor must differ between conditions")
            elif len(set(values)) != 1:
                raise ValueError(f"Calibration panel conditions differ outside contrast factor: {factor_name}")
        signatures = tuple(
            tuple(getattr(condition, factor_name) for factor_name in factor_names) for condition in self.conditions
        )
        if len(signatures) != len(set(signatures)):
            raise ValueError("Calibration panel conditions must have distinct factor signatures")
        return self


def validate_calibration_panel_v1(
    units: tuple[CalibrationUnitV1, ...],
    *,
    panel: CalibrationPanelV1,
) -> None:
    """Reject a calibration collection that does not cover one matched panel."""

    conditions = {condition.condition_id: condition for condition in panel.conditions}
    expected = set(product(conditions, panel.request_replicate_ids, panel.unit_ids))
    observed: Counter[tuple[str, str, str]] = Counter()
    task_factors: dict[tuple[str, str], set[tuple[str | None, str | None]]] = defaultdict(set)
    for row in units:
        if (
            row.benchmark == panel.benchmark
            and row.release_identity == panel.release_identity
            and row.model == panel.model
            and row.condition_id in conditions
            and row.replicate_id in panel.request_replicate_ids
            and row.unit_id in panel.unit_ids
        ):
            if row.condition_id is None or row.replicate_id is None:
                raise ValueError("Calibration panel rows require condition and request-replicate identifiers")
            expected_condition = conditions[row.condition_id]
            mismatches = tuple(
                factor_name
                for factor_name in CalibrationConditionV1.model_fields
                if factor_name != "condition_id"
                and getattr(row, factor_name) != getattr(expected_condition, factor_name)
            )
            if mismatches:
                raise ValueError(
                    f"Calibration panel {panel.panel_id!r} condition {row.condition_id!r} "
                    f"has undeclared factor values: {mismatches}"
                )
            observed[(row.condition_id, row.replicate_id, row.unit_id)] += 1
            task_factors[(row.replicate_id, row.unit_id)].add((row.analysis_specification, row.stream))
    duplicates = tuple(sorted(key for key, count in observed.items() if count != 1))
    if duplicates:
        raise ValueError(f"Calibration panel {panel.panel_id!r} contains duplicate requests: {duplicates}")
    missing = tuple(sorted(expected - observed.keys()))
    if missing:
        raise ValueError(f"Calibration panel {panel.panel_id!r} is missing requests: {missing}")
    unmatched_tasks = tuple(sorted(key for key, values in task_factors.items() if len(values) != 1))
    if unmatched_tasks:
        raise ValueError(
            f"Calibration panel {panel.panel_id!r} changes task factors within matched units: {unmatched_tasks}"
        )


def _run_roots(sources: tuple[Path, ...]) -> tuple[Path, ...]:
    roots: set[Path] = set()
    for source in sources:
        root = source.resolve(strict=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Calibration source must be a regular directory: {root}")
        if (root / "run_config.json").is_file() or (root / "invocation.json").is_file():
            roots.add(root)
            continue
        roots.update(path.parent.resolve() for path in root.rglob("run_config.json"))
        roots.update(path.parent.resolve() for path in root.rglob("invocation.json"))
    if not roots:
        raise ValueError("Calibration sources contain no benchmark runs")
    return tuple(sorted(roots))


def _provider_metrics(run_root: Path) -> dict[str, tuple[int, int, int, int, float, float, int, int]]:
    grouped: dict[str, list[ProviderRequestEventV1]] = defaultdict(list)
    authoritative_paths, _ = provider_log_paths_v1(run_root)
    for path in authoritative_paths:
        for event in read_provider_terminal_events_v1(path):
            grouped[event.unit_id].append(event)
    metrics: dict[str, tuple[int, int, int, int, float, float, int, int]] = {}
    for unit_id, event_rows in grouped.items():
        events = tuple(event_rows)
        metrics[unit_id] = (
            len(events),
            sum(event.status == "failed" for event in events),
            sum(event.prompt_tokens for event in events),
            sum(event.completion_tokens for event in events),
            fsum(event.reported_cost_usd or 0.0 for event in events),
            fsum(event.elapsed_seconds or 0.0 for event in events),
            sum(event.usage_status == "invalid" for event in events),
            sum(event.usage_status == "not_reported" for event in events),
        )
    return metrics


def _token_accounting_status(
    response_count: int, failed_count: int, invalid_count: int, unreported_count: int
) -> TokenAccountingStatusV1:
    """Describe token totals for recorded requests; failed calls have unknown usage too."""
    if failed_count or invalid_count or unreported_count:
        return "partial"
    return "complete" if response_count else "unknown"


def _trialdev_turns_used(task_root: Path) -> int | None:
    paths = tuple(sorted(task_root.glob("conversation*.json")))
    if not paths:
        return None
    turns = 0
    for path in paths:
        conversation = read_json(path)
        if not isinstance(conversation, list):
            raise ValueError(f"TrialDev conversation is not a message array: {path}")
        for message in conversation:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise ValueError(f"TrialDev conversation contains an invalid message: {path}")
            turns += message["role"] == "assistant"
    return turns


def _trialeval_units(run_root: Path) -> list[CalibrationUnitV1]:
    """Project every scheduled TrialEval task and its provider telemetry."""

    config = read_json_model(TrialEvalRunConfigV1, run_root / "run_config.json")
    provider = _provider_metrics(run_root)
    rows: list[CalibrationUnitV1] = []
    for unit_id in config.task_ids:
        path = run_root / "items" / f"{unit_id}.json"
        record = read_json_model(TrialEvalItemResultV1, path) if path.is_file() else None
        scores = None if record is None else record.scores
        failures = () if scores is None else tuple(scores.outcome.failure_origins)
        execution_status: ExecutionStatusV1 = "not_started" if record is None else record.agent_output.status
        response_count, provider_failures, prompt, completion, cost, elapsed, invalid, unreported = provider.get(
            unit_id, (0, 0, 0, 0, 0.0, 0.0, 0, 0)
        )
        reasoning = config.experiment_condition.reasoning
        rows.append(
            CalibrationUnitV1(
                benchmark="trialeval",
                run_id=run_root.name,
                run_identity_sha256=config.run_identity_sha256,
                release_identity=config.participant_manifest_checksum,
                unit_id=unit_id,
                model=config.model,
                provider=config.routing.provider,
                upstream_provider=config.routing.openrouter_provider,
                reasoning_effort=reasoning.effort,
                decoding_seed=config.decoding.decoding_seed,
                condition_id=config.experiment_condition.condition_id,
                replicate_id=config.experiment_condition.request_replicate_id,
                procedure_assistance=config.experiment_condition.procedure_assistance,
                tool_choice=config.experiment_condition.tool_choice,
                memory_strategy=config.experiment_condition.memory_strategy,
                analysis_specification=config.task_evidence_factors[unit_id].analysis_specification,
                turn_limit=config.experiment_condition.maximum_turns_per_step,
                completion_token_limit=config.decoding.max_tokens,
                context_character_limit=config.maximum_context_characters,
                send_temperature=config.decoding.send_temperature,
                request_timeout_seconds=config.routing.request_timeout_seconds,
                watchdog_seconds=config.item_watchdog_seconds,
                provider_maximum_attempts=config.retry.maximum_attempts,
                provider_retry_base_delay_seconds=config.retry.base_delay_seconds,
                reported_cost_stop_usd=None,
                executor_image=f"{config.executor.image_reference}@{config.executor.image_id}",
                executor_limits_sha256=canonical_payload_sha256(config.executor.limits.model_dump(mode="json")),
                worker_count=config.workers,
                execution_status=execution_status,
                turns_used=None if record is None else record.agent_output.turns_used,
                primary_failure_origin=(failures[0] if failures else None),
                provider_response_count=response_count,
                provider_failure_count=provider_failures,
                provider_invalid_usage_count=invalid,
                provider_unreported_usage_count=unreported,
                token_accounting_status=_token_accounting_status(
                    response_count, provider_failures, invalid, unreported
                ),
                analysis_prompt_tokens=prompt,
                analysis_completion_tokens=completion,
                analysis_reported_cost_usd=cost,
                analysis_elapsed_seconds=elapsed,
            )
        )
    return rows


def _trialdev_units(run_root: Path) -> list[CalibrationUnitV1]:
    invocation = read_json_model(TrialDevRunInvocationV1, run_root / "invocation.json")
    if invocation.checksum is None:
        raise ValueError("TrialDev invocation has no content identity")
    provider = _provider_metrics(run_root)
    rows: list[CalibrationUnitV1] = []
    for unit_id in invocation.scheduled_task_ids:
        task_root = run_root / "tasks" / unit_id
        path = task_root / "run.json"
        run = read_json_model(TrialDevTaskRunV1, path) if path.is_file() else None
        # Numerical recovery and development value are continuous ranking outcomes,
        # not a binary certificate of a preferred analysis or decision method.
        execution_status: ExecutionStatusV1 = (
            ("incomplete_run" if task_root.exists() else "not_started") if run is None else run.execution_status
        )
        response_count, provider_failures, prompt, completion, cost, elapsed, invalid, unreported = provider.get(
            unit_id, (0, 0, 0, 0, 0.0, 0.0, 0, 0)
        )
        rows.append(
            CalibrationUnitV1(
                benchmark="trialdev",
                run_id=run_root.name,
                run_identity_sha256=invocation.checksum,
                release_identity=invocation.release_id,
                unit_id=unit_id,
                model=invocation.model,
                provider=invocation.provider,
                upstream_provider=invocation.openrouter_provider,
                reasoning_effort=invocation.experiment_condition.reasoning.effort,
                decoding_seed=invocation.decoding_seed,
                condition_id=invocation.experiment_condition.condition_id,
                replicate_id=invocation.experiment_condition.request_replicate_id,
                procedure_assistance=invocation.experiment_condition.procedure_assistance,
                tool_choice=invocation.experiment_condition.tool_choice,
                memory_strategy=invocation.experiment_condition.memory_strategy,
                stream=invocation.scheduled_task_streams[unit_id].value,
                turn_limit=invocation.experiment_condition.maximum_turns_per_step,
                completion_token_limit=invocation.maximum_completion_tokens,
                context_character_limit=invocation.maximum_context_characters,
                send_temperature=invocation.send_temperature,
                request_timeout_seconds=invocation.request_timeout_seconds,
                watchdog_seconds=invocation.watchdog_seconds,
                provider_maximum_attempts=invocation.retry.maximum_attempts,
                provider_retry_base_delay_seconds=invocation.retry.base_delay_seconds,
                reported_cost_stop_usd=invocation.reported_cost_stop_usd,
                executor_image=invocation.executor_image,
                executor_limits_sha256=canonical_payload_sha256(invocation.executor_limits.model_dump(mode="json")),
                worker_count=invocation.workers or 1,
                execution_status=execution_status,
                turns_used=_trialdev_turns_used(task_root),
                primary_failure_origin=(
                    execution_status
                    if execution_status in {"model_noncompletion", "evaluator_error", "incomplete_run"}
                    else None
                ),
                provider_response_count=response_count,
                provider_failure_count=provider_failures,
                provider_invalid_usage_count=invalid,
                provider_unreported_usage_count=unreported,
                token_accounting_status=_token_accounting_status(
                    response_count, provider_failures, invalid, unreported
                ),
                analysis_prompt_tokens=prompt,
                analysis_completion_tokens=completion,
                analysis_reported_cost_usd=cost,
                analysis_elapsed_seconds=elapsed,
            )
        )
    return rows


def calibration_cells_v1(units: tuple[CalibrationUnitV1, ...]) -> tuple[CalibrationCellV1, ...]:
    """Aggregate calibration units without dropping incomplete schedules."""

    grouped: dict[tuple[object, ...], list[CalibrationUnitV1]] = defaultdict(list)
    for unit in units:
        grouped[
            (
                unit.benchmark,
                unit.release_identity,
                unit.model,
                unit.provider,
                unit.upstream_provider,
                unit.reasoning_effort,
                unit.decoding_seed,
                unit.procedure_assistance,
                unit.tool_choice,
                unit.memory_strategy,
                unit.analysis_specification,
                unit.stream,
                unit.turn_limit,
                unit.completion_token_limit,
                unit.context_character_limit,
                unit.send_temperature,
                unit.request_timeout_seconds,
                unit.watchdog_seconds,
                unit.provider_maximum_attempts,
                unit.provider_retry_base_delay_seconds,
                unit.reported_cost_stop_usd,
                unit.executor_image,
                unit.executor_limits_sha256,
                unit.worker_count,
            )
        ].append(unit)
    cells: list[CalibrationCellV1] = []
    for factors, rows in sorted(grouped.items(), key=lambda item: tuple(str(value or "") for value in item[0])):
        (
            benchmark,
            release_identity,
            model,
            provider,
            upstream_provider,
            effort,
            decoding_seed,
            procedure_assistance,
            tool_choice,
            memory_strategy,
            analysis_specification,
            stream,
            turn_limit,
            completion_token_limit,
            context_character_limit,
            send_temperature,
            request_timeout_seconds,
            watchdog_seconds,
            provider_maximum_attempts,
            provider_retry_base_delay_seconds,
            reported_cost_stop_usd,
            executor_image,
            executor_limits_sha256,
            worker_count,
        ) = factors
        by_independent_unit: dict[tuple[str, str], list[CalibrationUnitV1]] = defaultdict(list)
        for row in rows:
            by_independent_unit[(row.release_identity, row.unit_id)].append(row)
        cells.append(
            CalibrationCellV1(
                benchmark=benchmark,
                release_identity=release_identity,
                model=model,
                provider=provider,
                upstream_provider=upstream_provider,
                reasoning_effort=effort,
                decoding_seed=decoding_seed,
                condition_ids=tuple(sorted({row.condition_id or "unspecified" for row in rows})),
                procedure_assistance=procedure_assistance,
                tool_choice=tool_choice,
                memory_strategy=memory_strategy,
                analysis_specification=analysis_specification,
                stream=stream,
                turn_limit=turn_limit,
                completion_token_limit=completion_token_limit,
                context_character_limit=context_character_limit,
                send_temperature=send_temperature,
                request_timeout_seconds=request_timeout_seconds,
                watchdog_seconds=watchdog_seconds,
                provider_maximum_attempts=provider_maximum_attempts,
                provider_retry_base_delay_seconds=provider_retry_base_delay_seconds,
                reported_cost_stop_usd=reported_cost_stop_usd,
                executor_image=executor_image,
                executor_limits_sha256=executor_limits_sha256,
                worker_count=worker_count,
                request_replicate_ids=tuple(sorted({row.replicate_id or "unspecified" for row in rows})),
                scheduled_requests=len(rows),
                independent_unit_count=len(by_independent_unit),
                execution_completed=sum(row.execution_status in {"success", "completed"} for row in rows),
                model_noncompletion=sum(
                    row.execution_status in {"max_turns_reached", "model_noncompletion"} for row in rows
                ),
                evaluator_error=sum(row.execution_status == "evaluator_error" for row in rows),
                provider_failures=sum(row.provider_failure_count for row in rows),
                provider_invalid_usage_count=_known_count_sum(tuple(row.provider_invalid_usage_count for row in rows)),
                provider_unreported_usage_count=_known_count_sum(
                    tuple(row.provider_unreported_usage_count for row in rows)
                ),
                token_accounting_status=(
                    "partial"
                    if any(row.token_accounting_status == "partial" for row in rows)
                    else "complete" if all(row.token_accounting_status == "complete" for row in rows) else "unknown"
                ),
                analysis_tokens=sum(row.analysis_prompt_tokens + row.analysis_completion_tokens for row in rows),
                observed_cost_usd=fsum(row.analysis_reported_cost_usd for row in rows),
                observed_elapsed_seconds=fsum(row.analysis_elapsed_seconds for row in rows),
            )
        )
    return tuple(cells)


def _known_count_sum(counts: tuple[int | None, ...]) -> int | None:
    """Return unknown if any contributing count is unknown."""
    return None if any(count is None for count in counts) else sum(count for count in counts if count is not None)


def summarize_calibration_runs_v1(sources: tuple[Path, ...]) -> CalibrationSummaryV1:
    """Summarize every prospective unit in one or more run trees."""

    units: list[CalibrationUnitV1] = []
    for run_root in _run_roots(sources):
        if (run_root / "run_config.json").is_file():
            units.extend(_trialeval_units(run_root))
        else:
            units.extend(_trialdev_units(run_root))
    ordered = tuple(sorted(units, key=lambda row: (row.run_identity_sha256, row.unit_id)))
    return CalibrationSummaryV1(
        units=ordered,
        cells=calibration_cells_v1(ordered),
        run_count=len({row.run_identity_sha256 for row in ordered}),
        scheduled_request_count=len(ordered),
        independent_unit_count=len({(row.release_identity, row.unit_id) for row in ordered}),
    ).with_checksum()


__all__ = [
    "CalibrationCellV1",
    "CalibrationConditionV1",
    "CalibrationPanelV1",
    "CalibrationSummaryV1",
    "CalibrationUnitV1",
    "calibration_cells_v1",
    "summarize_calibration_runs_v1",
    "validate_calibration_panel_v1",
]
