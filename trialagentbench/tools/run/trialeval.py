"""Run TrialEval against a benchmark release surface."""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue
from trialagentbench_verifier.assets import read_asset_catalog
from trialagentbench_verifier.execution_contracts import AnalysisMode, ReportingInterfaceV1
from trialagentbench_verifier.executor import resolve_trusted_executor_environment
from trialagentbench_verifier.run_contracts import ProviderUnavailabilityReceiptV1

from trialagentbench.adapters import ProviderRouting, close_provider, get_provider
from trialagentbench.contracts.core.config import (
    DecodingConfigV1,
    ProcedureAssistanceV1,
    ProviderRetryPolicyV1,
    ReasoningEffortV1,
    RoutingConfigV1,
    TrialEvalAgentStatusV1,
)
from trialagentbench.contracts.core.runs import (
    ProviderRequestEventV1,
    ProviderTelemetrySummaryV1,
    TrialEvalAgentOutputV1,
    TrialEvalItemResultV1,
    TrialEvalRunConfigV1,
)
from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
)
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.trace.observable import (
    BenchmarkRuntimeFailureTypeV1,
    BenchmarkRuntimeTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench.execution_policy import (
    TRIALEVAL_DEFAULT_WORKERS,
    TRIALEVAL_RELEASE_BUDGET_V1,
)
from trialagentbench.grading.outcome import grade_trialeval_outcome_v1
from trialagentbench.io import read_json_model
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.io.exclusive_publication import atomic_text_writer
from trialagentbench.io.json import append_jsonl_model, write_json_model
from trialagentbench.ports.llm_provider import ProviderUnavailableError
from trialagentbench.release import load_participant_release
from trialagentbench.trialeval.agent import DEFAULT_MAX_TURNS, condition_provenance_v1, run_agent
from trialagentbench.trialeval.conditions import prompt_set_sha256_v1
from trialagentbench.trialeval.data import discover_participant_items, participant_task_factors
from trialagentbench.trialeval.schema import BenchmarkItem
from trialagentbench.util.campaign_control import CAMPAIGN_BUDGET_SCOPE, CampaignCapacityStop, exhausted_budgets
from trialagentbench.util.campaign_dispatch import execution_slice
from trialagentbench.util.experiment_condition import (
    resolve_experiment_condition_v1,
)
from trialagentbench.util.parallel import run_bounded_tasks
from trialagentbench.util.provider_environment import load_provider_dotenv
from trialagentbench.util.provider_telemetry import read_provider_request_log_v1, summarize_provider_telemetry_v1
from trialagentbench.util.reported_cost import RunStopSignal, StoppableProvider
from trialagentbench.util.run_coverage import (
    load_or_create_run_coverage_v1,
    reconcile_run_coverage_v1,
)
from trialagentbench.util.run_ownership import exclusive_run_directory
from trialagentbench.util.source_identity import (
    harness_source_sha256_v1,
    source_file_sha256_v1,
)


def _selected_task_ids(
    contexts: Mapping[str, str], selectors: Sequence[str], *, context: str | None = None
) -> list[str]:
    """Resolve canonical evidence contexts identically for preview and execution."""
    selected_task_ids = list(selectors) if selectors else list(contexts)
    if len(set(selected_task_ids)) != len(selected_task_ids):
        raise ValueError("--task-id values must be unique.")
    unknown_task_ids = sorted(set(selected_task_ids) - set(contexts))
    if unknown_task_ids:
        raise ValueError(f"Requested task IDs are absent from the participant release: {unknown_task_ids}")
    context = context or ("all" if selectors else "C5")
    if context != "all":
        selected_task_ids = [task_id for task_id in selected_task_ids if contexts[task_id] == context]
        if selectors and len(selected_task_ids) != len(selectors):
            raise ValueError("Explicit task IDs conflict with the requested evidence context.")
    if not selected_task_ids:
        raise ValueError("No TrialEval tasks match the requested evidence context.")
    return selected_task_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--campaign-job", help=argparse.SUPPRESS)
    parser.add_argument("--dispatch-task-id", action="append", help=argparse.SUPPRESS)
    parser.add_argument("--participant-release", type=Path, required=True, help="TrialEval participant release ZIP.")
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="Trusted SHA-256 of the participant release ZIP from the release-set manifest.",
    )
    parser.add_argument(
        "--task-id",
        nargs="+",
        help="Exact opaque task IDs. Omit to run the canonical C5 panel across all released scientific variants.",
    )
    parser.add_argument(
        "--context",
        choices=("C1", "C2", "C3", "C4", "C5", "all"),
        help="Evidence presentation. Defaults to C5 unless exact task IDs are supplied.",
    )
    parser.add_argument("--model", required=True, help="Agent model identifier.")
    parser.add_argument(
        "--executor-image",
        help="Local reference for the qualified executor image; the immutable expected ID remains authoritative.",
    )
    parser.add_argument(
        "--expected-executor-image-id",
        required=True,
        help="Immutable sha256: Docker image ID from this host's authenticated executor attestation.",
    )
    parser.add_argument(
        "--expected-executor-environment-sha256",
        required=True,
        help="Externally trusted executor inventory SHA-256 from the release-set manifest.",
    )
    parser.add_argument(
        "--procedure-assistance",
        choices=("output_contract_only", "unordered_checklist"),
        default="output_contract_only",
        help="Analysis instructions: the clean output contract or an unordered checklist ablation.",
    )
    parser.add_argument(
        "--reporting-interface",
        choices=("v1", "clarified"),
        default="v1",
        help="Reporting interface: the 1.0 wording and helpers (v1) or the clarified helpers with repair turns.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("openai", "openai_responses", "openrouter", "offline_control"),
        help=(
            "Exact API transport: openai uses Chat Completions, openai_responses "
            "uses Responses, and openrouter uses pinned OpenRouter Chat Completions."
        ),
    )
    parser.add_argument(
        "--memory",
        choices=("bounded", "openai_responses_compaction"),
        default="bounded",
        help=(
            "Conversation memory strategy. OpenAI Responses compaction requires "
            "--provider openai_responses and records its additional token usage."
        ),
    )
    parser.add_argument("--turns", type=int, help="Maximum agent turns; context defaults apply when omitted.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
        help="Maximum completion tokens per provider turn.",
    )
    parser.add_argument("--analysis-language", choices=("python_r", "python", "r"), default="python_r")
    parser.add_argument(
        "--condition-id",
        default="primary",
        help="Stable identifier for this experimental condition.",
    )
    parser.add_argument(
        "--request-replicate-id",
        default="request-1",
        help="Stable identifier for this provider request replicate.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Exact provider-supported reasoning effort; omitted by default.",
    )
    parser.add_argument(
        "--exclude-returned-reasoning",
        action="store_true",
        help="Ask the route not to return reasoning; by default returned reasoning is kept in the provider log.",
    )
    parser.add_argument(
        "--replay-returned-reasoning",
        action="store_true",
        help="Send returned reasoning details back on later turns (OpenRouter reasoning-state continuity).",
    )
    parser.add_argument(
        "--provider-capability-snapshot",
        type=Path,
        default=None,
        help="Source-bound model context/completion/reasoning capacities; required for explicit reasoning effort.",
    )
    parser.add_argument(
        "--item-watchdog-seconds",
        type=int,
        default=TRIALEVAL_RELEASE_BUDGET_V1.wall_time_limit_seconds,
        help="Per-item wall-clock budget across all turns and tools (default: 3600 seconds).",
    )
    parser.add_argument("--output-dir", default="results", help="Parent directory for the timestamped run.")
    parser.add_argument(
        "--resume-run-dir",
        "--append-run-dir",
        dest="resume_run_dir",
        help="Resume pending items in an identity-matched run with a frozen prospective schedule.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=TRIALEVAL_DEFAULT_WORKERS,
        help="Concurrent item executions.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress turn-level output.")
    parser.add_argument(
        "--openrouter-provider",
        help="Required exact upstream-provider pin when --provider=openrouter.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=TRIALEVAL_RELEASE_BUDGET_V1.provider_request_timeout_seconds,
        help="Per-attempt provider request timeout (default: 300 seconds).",
    )
    parser.add_argument("--provider-max-attempts", type=int, default=5)
    parser.add_argument("--provider-retry-base-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--provider-unavailability-policy", choices=("charge_all_v1", "defer_unanswered_v1"), default="charge_all_v1"
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send temperature when the selected endpoint does not support it.",
    )
    parser.add_argument(
        "--decoding-seed",
        type=int,
        help="Optional non-negative provider decoding seed.",
    )
    parser.add_argument("--dotenv", action="store_true", help="Load provider credentials from .env.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze the validated invocation and task schedule without contacting a provider; use resume to execute.",
    )
    return parser


def _provider_unavailability_receipt(event: ProviderRequestEventV1) -> ProviderUnavailabilityReceiptV1:
    """Retain the actual terminal request, never infer an outage from a generic failure."""
    payload = event.model_dump(mode="json")
    return ProviderUnavailabilityReceiptV1(
        request_event=payload,
        request_event_sha256=canonical_payload_sha256(payload),
    )


def _save_item_result(
    *,
    output_dir: Path,
    item_id: str,
    agent_output: dict[str, JsonValue],
    run_config: TrialEvalRunConfigV1,
) -> None:
    items_dir = output_dir / "items"
    raw_status = agent_output.get("status")
    if raw_status not in {"success", "max_turns_reached", "failed"}:
        raise ValueError(f"Unexpected TrialEval agent status: {raw_status!r}.")
    agent = TrialEvalAgentOutputV1(
        status=cast(TrialEvalAgentStatusV1, raw_status),
        failure_type=agent_output.get("failure_type"),
        provider_unavailability=agent_output.get("provider_unavailability"),
        turns_used=_integer(agent_output.get("turns_used")),
        report=str(agent_output.get("report")) if agent_output.get("report") is not None else None,
        condition_provenance=agent_output.get("condition_provenance"),
    )
    result = TrialEvalItemResultV1(
        item_id=item_id,
        timestamp_utc=datetime.now(UTC),
        run_config=run_config,
        agent_output=agent,
    )
    conversation = agent_output.get("conversation")
    if isinstance(conversation, list) and conversation:
        logs_dir = output_dir / "logs"
        with atomic_text_writer(logs_dir / f"{item_id}_conversation.json") as handle:
            handle.write(json.dumps(conversation, indent=2, default=str, sort_keys=True) + "\n")
    events = agent_output.get("events")
    event_path = output_dir / "logs" / f"{item_id}_events.jsonl"
    if isinstance(events, list) and events:
        logs_dir = output_dir / "logs"
        with atomic_text_writer(logs_dir / f"{item_id}_events.jsonl") as handle:
            handle.write("".join(json.dumps(event, sort_keys=True) + "\n" for event in events))
    elif agent.status == "failed" and event_path.is_file():
        retained = [
            BenchmarkRuntimeTraceEventV1.model_validate_json(line) for line in event_path.read_text().splitlines()
        ]
        if retained and not any(event.event_type == "step_terminal" for event in retained):
            # A killed process cannot close its trace. Publish the retained
            # infrastructure failure before making its recovered item visible.
            payload = runtime_event_source_payload_v1(
                benchmark="trialeval",
                task_id=item_id,
                program_id=None,
                scenario_id=None,
                objective_id=None,
                phase_id="task",
                step_id="analysis",
                event_type="step_terminal",
                terminal_status="failed",
                failure_type=agent.failure_type,
                conversation_message=None,
            )
            terminal = retained[0].model_copy(
                update={
                    "event_id": f"trialeval:{item_id}:{len(retained):06d}",
                    "event_index": len(retained),
                    "timestamp": datetime.now(UTC),
                    "event_type": "step_terminal",
                    "terminal_status": "failed",
                    "failure_type": agent.failure_type,
                    "source_payload_sha256": canonical_payload_sha256(payload),
                }
            )
            append_jsonl_model(event_path, terminal)
    # Completion is visible only after the supporting logs are durably published.
    _write_json_model_exclusive(items_dir / f"{item_id}.json", result)


def _write_json_model_exclusive(path: Path, model: TrialEvalItemResultV1) -> None:
    """Durably create one immutable checkpoint without overwrite races."""

    payload = json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    with atomic_text_writer(path, exclusive=True) as handle:
        handle.write(payload)


def _archive_partial_item(
    output_dir: Path, item_id: str, *, candidate_paths: tuple[Path, ...] | None = None
) -> Path | None:
    """Move raw incomplete-attempt logs aside without matching active telemetry discovery."""

    candidates = (
        candidate_paths
        if candidate_paths is not None
        else (
            *sorted((output_dir / "logs").glob(f"{item_id}_provider_*.jsonl")),
            output_dir / "logs" / f"{item_id}_conversation.json",
            output_dir / "logs" / f"{item_id}_events.jsonl",
            output_dir / "logs" / f"{item_id}_workspace",
        )
    )
    present = tuple(path for path in candidates if path.exists())
    if not present:
        return None
    attempt_root = output_dir / "failed_attempts" / item_id
    attempt_index = 1
    while (attempt_root / f"attempt-{attempt_index}").exists():
        attempt_index += 1
    archive = attempt_root / f"attempt-{attempt_index}"
    archive.mkdir(parents=True)
    for path in present:
        if path.is_dir():
            path.rename(archive / f"{path.name}.archived")
        else:
            suffix = ".jsonl" if path.suffix == ".jsonl" else ".json"
            path.rename(archive / f"{path.stem}.archived{suffix}")
    return archive


def _integer(value: JsonValue) -> int:
    """Return an explicitly persisted integer."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Expected persisted integer, observed {type(value).__name__}.")


def _run_materialized(
    args: argparse.Namespace,
    *,
    participant_dir: Path,
    participant_archive_sha256: str,
    participant_manifest_checksum: str,
) -> int:
    with ExitStack() as ownership:
        return _execute_materialized(
            args,
            participant_dir=participant_dir,
            participant_archive_sha256=participant_archive_sha256,
            participant_manifest_checksum=participant_manifest_checksum,
            ownership=ownership,
        )


def _execute_materialized(
    args: argparse.Namespace,
    *,
    participant_dir: Path,
    participant_archive_sha256: str,
    participant_manifest_checksum: str,
    ownership: ExitStack,
) -> int:
    """Execute a TrialEval run from one verified materialized participant archive."""

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    max_turns = int(args.turns) if args.turns is not None else DEFAULT_MAX_TURNS
    if max_turns < 1:
        raise ValueError("--turns must be at least 1")
    if args.item_watchdog_seconds < 1:
        raise ValueError("--item-watchdog-seconds must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be at least 1")
    if args.provider == "openai_responses" and args.decoding_seed is not None:
        raise ValueError("--decoding-seed is not supported by --provider openai_responses")
    experiment_condition = resolve_experiment_condition_v1(
        condition_id=args.condition_id,
        analysis_language=cast(AnalysisMode, args.analysis_language),
        request_replicate_id=args.request_replicate_id,
        reasoning_effort=cast(ReasoningEffortV1 | None, args.reasoning_effort),
        provider_capability_snapshot=args.provider_capability_snapshot,
        provider=args.provider,
        model=args.model,
        openrouter_provider=args.openrouter_provider or None,
        procedure_assistance=cast(ProcedureAssistanceV1, args.procedure_assistance),
        reporting_interface=cast(ReportingInterfaceV1, args.reporting_interface),
        maximum_turns_per_step=max_turns,
        tool_choice="auto",
        memory_strategy=args.memory,
        exclude_reasoning=args.exclude_returned_reasoning,
        replay_reasoning=args.replay_returned_reasoning,
    )
    if args.dotenv:
        load_provider_dotenv()

    declared_task_ids, declared_factors = participant_task_factors(participant_dir)
    selected_task_ids = _selected_task_ids(
        {task_id: declared_factors[task_id].context_configuration for task_id in declared_task_ids},
        args.task_id or (),
        context=args.context,
    )
    item_by_id = discover_participant_items(participant_dir, task_ids=tuple(selected_task_ids))
    items = [item_by_id[task_id] for task_id in selected_task_ids]
    if not items:
        raise ValueError("The participant release contains no selected benchmark items.")

    timestamp = datetime.now(UTC)
    model_slug = args.model.replace("/", "_").replace(".", "_")
    if args.resume_run_dir:
        output_dir = Path(args.resume_run_dir).resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Cannot resume missing TrialEval run directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir) / model_slug / timestamp.strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
    ownership.enter_context(exclusive_run_directory(output_dir))
    executor = resolve_trusted_executor_environment(
        image=args.executor_image,
        expected_image_id=args.expected_executor_image_id,
        expected_environment_sha256=args.expected_executor_environment_sha256,
        limits=TRIALEVAL_RELEASE_BUDGET_V1.code_execution,
    )
    config_payload: dict[str, JsonValue] = {
        "schema_id": "trialagentbench_trialeval_run_config_v1",
        "schema_version": 1,
        "timestamp_utc": timestamp.isoformat(),
        "model": args.model,
        "budget_profile_id": TRIALEVAL_RELEASE_BUDGET_V1.profile_id,
        "item_watchdog_seconds": args.item_watchdog_seconds,
        "maximum_context_characters": TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters,
        "experiment_condition": experiment_condition.model_dump(mode="json"),
        "decoding": DecodingConfigV1(
            temperature=0.0,
            max_tokens=args.max_tokens,
            send_temperature=not args.omit_temperature,
            decoding_seed=args.decoding_seed,
        ).model_dump(mode="json"),
        "routing": RoutingConfigV1(
            provider=args.provider,
            openrouter_provider=args.openrouter_provider,
            request_timeout_seconds=args.request_timeout_seconds,
        ).model_dump(mode="json"),
        "retry": ProviderRetryPolicyV1(
            maximum_attempts=args.provider_max_attempts,
            base_delay_seconds=args.provider_retry_base_delay_seconds,
            unavailability_policy=args.provider_unavailability_policy,
        ).model_dump(mode="json"),
        "executor": executor.model_dump(mode="json"),
        "participant_archive": Path(args.participant_release).name,
        "participant_archive_sha256": participant_archive_sha256,
        "participant_manifest_checksum": participant_manifest_checksum,
        "prompt_set_sha256": prompt_set_sha256_v1(),
        "scorer_source_sha256": source_file_sha256_v1(grade_trialeval_outcome_v1),
        "harness_source_sha256": harness_source_sha256_v1(),
        "task_evidence_factors": {
            task_id: declared_factors[task_id].model_dump(mode="json") for task_id in selected_task_ids
        },
        "task_ids": cast(list[JsonValue], selected_task_ids),
        "workers": args.workers,
        "n_items": len(items),
    }
    proposed_config = TrialEvalRunConfigV1.create(**config_payload)
    if args.resume_run_dir:
        run_config = read_json_model(TrialEvalRunConfigV1, output_dir / "run_config.json")
        if run_config.run_identity_sha256 != proposed_config.run_identity_sha256:
            raise ValueError("Resume arguments or source content conflict with the persisted TrialEval identity.")
        if run_config.task_ids != selected_task_ids:
            raise ValueError("Resume selection conflicts with the immutable TrialEval denominator.")
    else:
        run_config = proposed_config
        write_json_model(output_dir / "run_config.json", run_config)

    coverage_path = output_dir / "coverage.json"
    coverage = load_or_create_run_coverage_v1(
        coverage_path,
        run_identity_sha256=run_config.run_identity_sha256,
        unit_ids=tuple(run_config.task_ids),
        resume=bool(args.resume_run_dir),
    )

    existing_results: dict[str, TrialEvalItemResultV1] = {}
    for path in sorted((output_dir / "items").glob("*.json")):
        result = read_json_model(TrialEvalItemResultV1, path)
        if path.stem != result.item_id or result.item_id not in run_config.task_ids:
            raise ValueError(f"Persisted TrialEval checkpoint has an unknown identity: {path}")
        if result.run_config != run_config:
            raise ValueError(f"Persisted TrialEval checkpoint has run identity drift: {result.item_id}")
        if result.item_id in existing_results:
            raise ValueError(f"Duplicate TrialEval checkpoint identity: {result.item_id}")
        existing_results[result.item_id] = result
    checkpoint_ids = tuple(task_id for task_id in run_config.task_ids if task_id in existing_results)
    if any(task_id not in existing_results for task_id in coverage.completed_unit_ids):
        raise ValueError("TrialEval coverage claims completion without an immutable item checkpoint.")
    if coverage.completed_unit_ids != checkpoint_ids:
        coverage = reconcile_run_coverage_v1(
            coverage_path,
            coverage,
            completed_unit_ids=checkpoint_ids,
        )
    if (output_dir / "provider_telemetry_summary.json").exists():
        telemetry = read_json_model(ProviderTelemetrySummaryV1, output_dir / "provider_telemetry_summary.json")
        if len(existing_results) != len(run_config.task_ids):
            raise ValueError("Completed TrialEval marker exists with incomplete schedule coverage.")
        if (
            telemetry.run_identity_sha256 != coverage.run_identity_sha256
            or telemetry.schedule_sha256 != coverage.schedule_sha256
            or telemetry.unit_ids != coverage.unit_ids
            or telemetry.completed_unit_ids != coverage.completed_unit_ids
        ):
            raise ValueError("Completed TrialEval telemetry does not match its run denominator.")
        print(f"Run already complete: {output_dir}")
        return 0

    if args.prepare_only:
        print(f"Prepared {len(run_config.task_ids)} tasks without provider execution: {output_dir}")
        return 0

    routing = ProviderRouting(provider=args.provider, openrouter_provider=args.openrouter_provider)
    from trialagentbench.util.campaign_control import request_control_for_run

    request_control = request_control_for_run(
        campaign_root=args.campaign_root, job_id=args.campaign_job, run_root=output_dir
    )
    slice_ids = execution_slice(
        campaign_root=args.campaign_root,
        job_id=args.campaign_job,
        run_root=output_dir,
        full_task_ids=run_config.task_ids,
        requested_task_ids=args.dispatch_task_id,
    )
    items = [item_by_id[task_id] for task_id in slice_ids]
    work: list[tuple[int, BenchmarkItem]] = []
    capacity_stop: CampaignCapacityStop | None = None
    for item in items:
        if item.item_id not in existing_results:
            if args.resume_run_dir:
                events: list[ProviderRequestEventV1] = []
                torn_bytes = 0
                for path in sorted((output_dir / "logs").glob(f"{item.item_id}_provider_responses*.jsonl")):
                    records, tail_bytes = read_provider_request_log_v1(path)
                    events.extend(records)
                    torn_bytes += tail_bytes
                if any(event.unit_id != item.item_id for event in events):
                    raise ValueError("Provider log identity differs from its TrialEval task")
                outcomes = [event for event in events if event.model_outcome is not None]
                if (
                    not outcomes
                    and request_control is not None
                    and request_control.model is not None
                    and events
                    and events[-1].exception_type == "CampaignCapacityStop"
                    and events[-1].failure_type == "cancelled"
                    and not events[-1].response_received
                ):
                    # An authenticated spending stop before transport leaves work
                    # pending for budget censoring, rather than charging a model
                    # outcome or manufacturing an infrastructure retry.
                    exhausted = exhausted_budgets(request_control.database, request_control.configuration)
                    scope = next(
                        (scope for scope in (CAMPAIGN_BUDGET_SCOPE, request_control.model.name) if scope in exhausted),
                        None,
                    )
                    if scope is not None:
                        capacity_stop = CampaignCapacityStop(
                            exhausted[scope], campaign_wide=scope == CAMPAIGN_BUDGET_SCOPE
                        )
                        continue
                interrupted = (
                    torn_bytes
                    or sum(event.status == "started" for event in events)
                    != sum(event.status != "started" for event in events)
                    or (events and events[-1].failure_type == "cancelled")
                )
                received = any(event.status == "succeeded" or event.response_received for event in events)
                recovery_failure: BenchmarkRuntimeFailureTypeV1 | None = None
                unavailable_receipt = None
                terminal_path = output_dir / "logs" / f"{item.item_id}_events.jsonl"
                runtime_terminals = (
                    [
                        event
                        for line in terminal_path.read_text().splitlines()
                        if (event := BenchmarkRuntimeTraceEventV1.model_validate_json(line)).event_type
                        == "step_terminal"
                    ]
                    if terminal_path.exists()
                    else []
                )
                if outcomes:
                    recovery_failure = "contract_error"
                elif (
                    events
                    and (
                        not runtime_terminals
                        or runtime_terminals[-1].failure_type in {None, "provider_error", "provider_unavailable"}
                    )
                    and events[-1].unavailability_policy == "defer_unanswered_v1"
                    and events[-1].exception_type == "ProviderUnavailableError"
                    and events[-1].status == "failed"
                    and events[-1].failure_type == "provider_error"
                    and not events[-1].response_received
                ):
                    unavailable_receipt = _provider_unavailability_receipt(events[-1])
                    recovery_failure = "provider_unavailable"
                elif runtime_terminals and runtime_terminals[-1].failure_type is not None:
                    # A recorded terminal ends this primary even when its origin
                    # is unknown. Missing telemetry cannot turn runtime_error
                    # into infrastructure eligibility or authorize a rerun.
                    recovery_failure = runtime_terminals[-1].failure_type
                elif events and events[-1].status == "failed" and events[-1].failure_type != "cancelled":
                    # The request already failed even if no response arrived
                    # and the process died before publishing its runtime
                    # terminal. Neither missing bytes nor an unknown failure
                    # origin permits another model attempt.
                    declared_timeout = (
                        events[-1].failure_type == "timeout" and events[-1].exception_type == "RuntimeDeadlineExceeded"
                    )
                    recovery_failure = "timeout" if declared_timeout else "provider_error"
                elif interrupted:
                    recovery_failure = "evaluator_error"
                elif received:
                    recovery_failure = "runtime_error"
                if recovery_failure is not None:
                    # Retain the primary. An unclean interruption is an
                    # infrastructure failure; only the campaign's separately
                    # labelled, eligibility-checked retry can replace it.
                    _save_item_result(
                        output_dir=output_dir,
                        item_id=item.item_id,
                        run_config=run_config,
                        agent_output={
                            "status": "failed",
                            "failure_type": recovery_failure,
                            **(
                                {"provider_unavailability": unavailable_receipt.model_dump(mode="json")}
                                if unavailable_receipt is not None
                                else {}
                            ),
                            "report": None,
                            "turns_used": max((event.turn_index for event in events), default=0),
                            "condition_provenance": condition_provenance_v1(
                                item,
                                max_turns=run_config.experiment_condition.maximum_turns_per_step,
                                procedure_assistance=run_config.experiment_condition.procedure_assistance,
                                analysis_specification=cast(
                                    TrialEvalAnalysisSpecificationV1, item.analysis_specification
                                ),
                                analysis_language=run_config.experiment_condition.analysis_language,
                                executor_limits=run_config.executor.limits,
                                reporting_interface=run_config.experiment_condition.reporting_interface,
                            ),
                        },
                    )
                    existing_results[item.item_id] = read_json_model(
                        TrialEvalItemResultV1, output_dir / "items" / f"{item.item_id}.json"
                    )
                    continue
                _archive_partial_item(output_dir, item.item_id)
            work.append((len(work), item))
    if set(existing_results) != set(coverage.completed_unit_ids):
        coverage = reconcile_run_coverage_v1(
            coverage_path,
            coverage,
            completed_unit_ids=tuple(task_id for task_id in run_config.task_ids if task_id in existing_results),
        )

    if capacity_stop is not None:
        raise capacity_stop

    stop = RunStopSignal()
    lock = threading.Lock()
    completed_indices: set[int] = set()
    completed_item_ids: set[str] = set(existing_results)

    def persist(index: int, item: BenchmarkItem, agent_output: dict[str, JsonValue]) -> None:
        _save_item_result(
            output_dir=output_dir,
            item_id=item.item_id,
            agent_output=agent_output,
            run_config=run_config,
        )
        with lock:
            completed_indices.add(index)
            completed_item_ids.add(item.item_id)
            completed_in_order = tuple(task_id for task_id in run_config.task_ids if task_id in completed_item_ids)
            reconcile_run_coverage_v1(coverage_path, coverage, completed_unit_ids=completed_in_order)
            if not args.quiet:
                print(f"[DONE {len(completed_item_ids)}/{len(run_config.task_ids)}] {item.item_id}")

    def process(index: int, item: BenchmarkItem) -> None:
        provider = get_provider(
            args.model,
            routing=routing,
            send_temperature=run_config.decoding.send_temperature,
            decoding_seed=run_config.decoding.decoding_seed,
            reasoning_effort=run_config.experiment_condition.reasoning.effort,
            exclude_reasoning=run_config.experiment_condition.reasoning.exclude_from_response,
            replay_reasoning=run_config.experiment_condition.reasoning.replay_in_context,
            timeout_s=run_config.routing.request_timeout_seconds,
            retry_policy=run_config.retry,
            memory_strategy=run_config.experiment_condition.memory_strategy,
            request_control=request_control,
        )
        provider = StoppableProvider(provider=provider, signal=stop)
        try:
            agent_output = cast(
                dict[str, JsonValue],
                run_agent(
                    item,
                    provider,
                    max_turns=run_config.experiment_condition.maximum_turns_per_step,
                    temperature=float(run_config.decoding.temperature),
                    max_tokens=int(run_config.decoding.max_tokens),
                    verbose=not args.quiet and args.workers == 1,
                    provider_log_path=output_dir / "logs" / f"{item.item_id}_provider_responses.jsonl",
                    conversation_log_path=output_dir / "logs" / f"{item.item_id}_conversation.json",
                    event_log_path=output_dir / "logs" / f"{item.item_id}_events.jsonl",
                    executor_image=run_config.executor.image_id,
                    executor_limits=run_config.executor.limits,
                    max_elapsed_seconds=float(run_config.item_watchdog_seconds),
                    max_context_chars=run_config.maximum_context_characters,
                    provider_capability=run_config.experiment_condition.provider_capability,
                    provider_unavailability_policy=run_config.retry.unavailability_policy,
                    analysis_language=run_config.experiment_condition.analysis_language,
                    procedure_assistance=run_config.experiment_condition.procedure_assistance,
                    reporting_interface=run_config.experiment_condition.reporting_interface,
                    analysis_specification=cast(
                        TrialEvalAnalysisSpecificationV1,
                        item.analysis_specification,
                    ),
                ),
            )
        except ProviderUnavailableError:
            # Eval has no mid-task conversation checkpoint. Retain this failed
            # primary; only the campaign's once-only eligible retry can replace it.
            log = output_dir / "logs" / f"{item.item_id}_provider_responses.jsonl"
            events, _ = read_provider_request_log_v1(log) if log.exists() else ([], 0)
            if not events:
                raise ValueError("Provider deferral has no retained request evidence") from None
            unavailable_receipt = _provider_unavailability_receipt(events[-1])
            persist(
                index,
                item,
                {
                    "status": "failed",
                    "failure_type": "provider_unavailable",
                    "provider_unavailability": unavailable_receipt.model_dump(mode="json"),
                    "report": None,
                    "turns_used": max((event.turn_index for event in events), default=0),
                    "condition_provenance": condition_provenance_v1(
                        item,
                        max_turns=run_config.experiment_condition.maximum_turns_per_step,
                        procedure_assistance=run_config.experiment_condition.procedure_assistance,
                        analysis_specification=cast(TrialEvalAnalysisSpecificationV1, item.analysis_specification),
                        analysis_language=run_config.experiment_condition.analysis_language,
                        executor_limits=run_config.executor.limits,
                        reporting_interface=run_config.experiment_condition.reporting_interface,
                    ),
                },
            )
            raise
        finally:
            close_provider(provider)
        persist(index, item, agent_output)

    run_bounded_tasks(
        work,
        lambda row: process(*row),
        workers=args.workers,
        consume=lambda _: None,
        stop=stop,
        request_control=request_control,
    )

    if len(completed_item_ids) != len(run_config.task_ids):
        print(f"Slice saved; {len(completed_item_ids)}/{len(run_config.task_ids)} full-panel items complete.")
        return 0
    summarize_provider_telemetry_v1(
        run_root=output_dir,
        coverage=load_or_create_run_coverage_v1(
            coverage_path,
            run_identity_sha256=run_config.run_identity_sha256,
            unit_ids=tuple(run_config.task_ids),
            resume=True,
        ),
    )
    print(f"Run artifacts saved to {output_dir}")
    print(f"Completed items: {len(completed_item_ids)}/{len(run_config.task_ids)}")
    print("Grade with trialagentbench campaign finish (campaigns) or trialagentbench grade trialeval (direct runs).")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a TrialEval run from an externally anchored participant ZIP."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    task_ids = None
    if args.participant_release.name.endswith(".assets.json"):
        catalog = read_asset_catalog(args.participant_release, expected_sha256=args.expected_sha256)
        if catalog.benchmark != "trialeval" or catalog.audience != "participant":
            raise ValueError("TrialEval execution requires participant assets")
        contexts = {task.task_id: task.context_tier for task in catalog.tasks}
        if any(context is None for context in contexts.values()):
            raise ValueError("TrialEval asset catalog omits evidence-context metadata")
        task_ids = tuple(
            _selected_task_ids(
                {task_id: str(context) for task_id, context in contexts.items()},
                args.task_id or (),
                context=args.context,
            )
        )
    with materialized_release_archive(
        args.participant_release,
        expected_sha256=args.expected_sha256,
        task_ids=task_ids,
    ) as materialized:
        participant = load_participant_release(materialized.root)
        return _run_materialized(
            args,
            participant_dir=participant.root,
            participant_archive_sha256=materialized.archive_sha256,
            participant_manifest_checksum=participant.manifest.checksum,
        )


if __name__ == "__main__":
    raise SystemExit(main())
