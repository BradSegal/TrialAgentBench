"""Consumer configuration and immutable custody for native benchmark campaigns."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, Decimal
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trialagentbench_verifier.execution_contracts import (
    DEFAULT_COMPLETION_TOKENS,
    EXECUTOR_SLOTS_PER_TASK,
    TRIALDEV_MAXIMUM_TURNS_PER_STEP,
    TRIALEVAL_MAXIMUM_TURNS,
    AnalysisMode,
    ReportingInterfaceV1,
)
from trialagentbench_verifier.io import canonical_sha256 as canonical_payload_sha256
from trialagentbench_verifier.run_contracts import ProviderRetryPolicyV1, ReasoningEffortV1
from trialagentbench_verifier.scorer import CurrentScorer

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
# Evaluator archives are public, so a route that can browse could retrieve
# answers. OpenRouter ``:online`` variants, Perplexity routes and search or
# deep-research models carry built-in retrieval and are never admitted.
WEB_ENABLED_ROUTE = re.compile(r":online$|^perplexity/|search", re.IGNORECASE)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CampaignRequestPricingV1(_Model):
    """Conservative text-route prices used to reserve each actual request."""

    prompt_usd_per_token: Decimal = Field(ge=0)
    completion_usd_per_token: Decimal = Field(ge=0)
    request_usd: Decimal = Field(ge=0)

    def reserve_micro_usd(self, *, prompt_tokens: int, completion_tokens: int) -> int:
        """Round the worst-case request charge upwards to integer microdollars."""
        if prompt_tokens < 0 or completion_tokens < 1:
            raise ValueError("Request token bounds must be nonnegative with a positive completion allowance")
        cost = (
            prompt_tokens * self.prompt_usd_per_token
            + completion_tokens * self.completion_usd_per_token
            + self.request_usd
        )
        return max(1, int((cost * 1_000_000).to_integral_value(rounding=ROUND_CEILING)))


class CampaignModelV1(_Model):
    """One exact model/route and decoding condition, with explicit commercial premises."""

    name: Identifier
    model: str = Field(min_length=1)
    provider: Literal["openrouter", "offline_control"] = "openrouter"
    upstream_provider: str | None = Field(default=None, min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    # Provider-returned reasoning is retained in the run's provider log by default; exclusion asks the route not to
    # return it. Replay sends returned reasoning details back on later turns (encrypted reasoning state).
    exclude_reasoning: bool = Field(default=False, exclude_if=lambda value: value is False)
    replay_reasoning: bool = Field(default=False, exclude_if=lambda value: value is False)
    capability_snapshot: Path | None = None
    maximum_completion_tokens: int = Field(default=DEFAULT_COMPLETION_TOKENS, ge=1)
    send_temperature: bool = False
    decoding_seed: int | None = Field(default=None, ge=0)
    request_timeout_seconds: float = Field(default=90, gt=0)
    retry: ProviderRetryPolicyV1 = Field(default_factory=ProviderRetryPolicyV1)
    spend_limit_micro_usd: int = Field(gt=0)
    reservation_micro_usd: int = Field(gt=0)
    reservation_tokens: int = Field(gt=0)
    pricing_premise: str = Field(min_length=1)
    request_pricing: CampaignRequestPricingV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)

    @field_validator("model")
    @classmethod
    def reject_web_enabled_route(cls, model: str) -> str:
        """Reject routes with built-in web retrieval."""
        if WEB_ENABLED_ROUTE.search(model):
            raise ValueError(f"Model route {model!r} has built-in web retrieval and cannot be benchmarked")
        return model

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        """Require one request to fit the declared funds and capacity."""
        if (self.provider == "openrouter") != (self.upstream_provider is not None):
            raise ValueError("An upstream provider is required exactly for OpenRouter")
        if self.provider == "offline_control":
            if self.model not in {"smoke-zero-python", "smoke-zero-r"}:
                raise ValueError("Offline controls support only smoke-zero-python and smoke-zero-r")
            if (
                self.reasoning_effort is not None
                or self.capability_snapshot is not None
                or self.exclude_reasoning
                or self.replay_reasoning
            ):
                raise ValueError("Offline controls have no model capability or reasoning configuration")
        if self.exclude_reasoning and self.replay_reasoning:
            raise ValueError("Excluded reasoning cannot be replayed")
        if self.reservation_micro_usd > self.spend_limit_micro_usd:
            raise ValueError("One request reservation exceeds the model spending limit")
        if self.reservation_tokens > self.tokens_per_minute:
            raise ValueError("One request reservation exceeds the token window")
        if self.reservation_tokens < self.maximum_completion_tokens:
            raise ValueError("Token reservation must cover prompt and maximum completion")
        return self


class CampaignPanelV1(_Model):
    """A prospective common panel; empty selectors use each benchmark's native default."""

    benchmarks: tuple[Literal["trialeval", "trialdev"], ...] = ("trialeval", "trialdev")
    trialeval_tasks: tuple[str, ...] = ()
    trialdev_tasks: tuple[str, ...] = ()
    repetitions: int = Field(default=1, ge=1)
    condition_id: Identifier = "primary"
    analysis_language: AnalysisMode = Field(default="python_r", exclude_if=lambda value: value == "python_r")
    procedure_assistance: Literal["output_contract_only", "unordered_checklist"] = "output_contract_only"
    trialeval_turns: int = Field(default=TRIALEVAL_MAXIMUM_TURNS, ge=1)
    trialdev_turns_per_step: int = Field(default=TRIALDEV_MAXIMUM_TURNS_PER_STEP, ge=1)
    watchdog_seconds: int | None = Field(default=None, ge=1)
    # The withheld split uses a privately held evaluator for leaderboard scoring.
    split: Literal["public", "withheld"] = Field(default="public", exclude_if=lambda value: value == "public")
    # TrialEval reporting interface. Absent means "v1" for every configuration, so no campaign changes
    # interface silently; "clarified" is opt-in and serialised explicitly.
    reporting_interface: ReportingInterfaceV1 = Field(default="v1", exclude_if=lambda value: value == "v1")

    @model_validator(mode="after")
    def validate_panel(self) -> Self:
        """Reject empty benchmark selection and duplicate selectors."""
        if not self.benchmarks or len(set(self.benchmarks)) != len(self.benchmarks):
            raise ValueError("Campaign benchmarks must be nonempty and unique")
        if any(len(set(rows)) != len(rows) for rows in (self.trialeval_tasks, self.trialdev_tasks)):
            raise ValueError("Campaign task selectors must be unique")
        if "trialdev" in self.benchmarks and self.procedure_assistance != "output_contract_only":
            raise ValueError(
                "TrialDev supports output-contract assistance only; declare a separate TrialEval ablation"
            )
        return self


class CampaignOperationsV1(_Model):
    """Recorded operational limits; changes require a newly prepared campaign."""

    model_workers: int = Field(default=1, ge=1)
    task_workers: int = Field(default=1, ge=1)
    executor_workers: int | None = Field(default=None, ge=1)
    grading_workers: int | None = Field(default=None, ge=1)
    spend_limit_micro_usd: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_nesting(self) -> Self:
        """Bound concurrent task-owned executors without multiplying worker limits."""
        if EXECUTOR_SLOTS_PER_TASK * self.model_workers * self.task_workers > self.executor_capacity:
            raise ValueError(
                "Two executor slots per active task are required for the analysis session and independent replay"
            )
        if self.grading_workers is not None and self.grading_workers > self.executor_capacity:
            raise ValueError("grading_workers must fit executor_workers")
        return self

    @property
    def executor_capacity(self) -> int:
        """Size the shared container limit from task concurrency unless overridden."""
        return self.executor_workers or EXECUTOR_SLOTS_PER_TASK * self.model_workers * self.task_workers


class CampaignAnalysisV1(_Model):
    """Prospective finite-panel comparison policy, implemented by the independent verifier."""

    bootstrap_repetitions: int = Field(default=2000, ge=100)
    seed: int = Field(default=65009, ge=0)
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    minimum_worlds_per_stratum: Literal[2] = 2
    multiplicity: Literal["holm_within_metric"] = "holm_within_metric"

    def panel_unavailable_reason(self, world_counts: tuple[int, ...], *, required_strata: int) -> str | None:
        """Withhold comparisons unless every required stratum has independent-world support.

        Parameters
        ----------
        world_counts
            Independent-world counts in each observed stratum, after collapsing linked tasks and repetitions.
        required_strata
            Number of strata required by the benchmark's analysis panel.

        Returns
        -------
        str or None
            The missing-stratum or insufficient-worlds reason, or ``None`` for supported panels.
        """
        if len(world_counts) < required_strata:
            return "missing_required_stratum"
        if any(count < self.minimum_worlds_per_stratum for count in world_counts):
            return "insufficient_independent_worlds"
        return None


class LeaderboardRowV1(_Model):
    """One model's headline on one benchmark track, with a world-clustered interval.

    ``bounded_recovery`` transforms the aggregate primary loss, with its
    availability, reversed confidence limits and delta-method standard error.
    Headlines require complete release strata
    and the declared independent-world minimum; unsupported panels carry an
    explicit unavailable reason. Complete-panel loss and
    development value are withheld (None) when any compared task lacks them.
    Intervals resample independent worlds within the benchmark's strata.
    ``standard_error`` is the primary bootstrap's standard error, or its
    delta-method transform for recovery; it is None when unavailable.

    ``scheduled_tasks`` is the paired panel: scheduled tasks that no model lost
    to budget censoring. ``budget_censored_tasks`` counts this model's own
    censored tasks on the benchmark; they are unscored and excluded, never
    scored as missing answers.
    """

    scorer: CurrentScorer

    benchmark: Literal["trialeval", "trialdev"]
    track: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    metric: Literal[
        "bounded_recovery",
        "mean_loss",
        "net_development_value_usd",
        "oracle_value_capture_pct",
        "decision_regret_musd",
        "consequential_error_rate",
        "material_decision_error_rate",
    ]
    direction: Literal["higher_is_better", "lower_is_better"]
    estimate: float | None
    interval_low: float | None = None
    interval_high: float | None = None
    standard_error: float | None = Field(
        default=None,
        ge=0,
        description=(
            "World-bootstrap standard error of the estimate, in the metric's unit: bounded recovery, mean loss, "
            "US dollars for net development value, US$ millions for decision regret, or a rate."
        ),
    )
    scheduled_tasks: int = Field(ge=1)
    scored_tasks: int = Field(ge=0)
    independent_worlds: int = Field(ge=1)
    budget_censored_tasks: int = Field(default=0, ge=0)
    population: Literal["full_release_strata", "partial_release_strata", "paired_completed_panel"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    missing_strata: tuple[tuple[str, str, int], ...] = Field(default=(), exclude_if=lambda value: not value)
    minimum_worlds_per_stratum: int | None = Field(default=None, ge=2, exclude_if=lambda value: value is None)
    unavailable_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    uncertainty_unavailable_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    warnings: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    nonpositive_oracle_worlds: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)

    @model_validator(mode="after")
    def validate_estimate(self) -> Self:
        """Require paired limits and a standard error that only accompany an estimate."""
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("Leaderboard intervals require both limits")
        if self.estimate is None and (self.interval_low is not None or self.standard_error is not None):
            raise ValueError("A withheld estimate cannot carry an interval or standard error")
        if self.scored_tasks > self.scheduled_tasks:
            raise ValueError("Scored tasks exceed the scheduled panel")
        return self


class CampaignTaskPriorityV1(_Model):
    """One prospective task priority; scheduling never changes the scientific panel."""

    benchmark: Literal["trialeval", "trialdev"]
    task_id: str = Field(min_length=1)


class CampaignDispatchV1(_Model):
    """Bounded, resumable execution slices within the full frozen native runs."""

    task_order: tuple[CampaignTaskPriorityV1, ...] = Field(min_length=1)
    tasks_per_slice: int = Field(default=1, ge=1)
    defer_seconds: float = Field(default=300, gt=0)
    maximum_provider_deferrals_per_task: int = Field(default=3, ge=1)
    maximum_consecutive_provider_deferrals: int = Field(default=3, ge=1)


class CampaignConfigurationV1(_Model):
    """One public configuration for preparation, execution, recovery and finishing."""

    schema_id: Literal["trialagentbench.campaign_configuration/v1"] = "trialagentbench.campaign_configuration/v1"
    release_directory: Path
    input_mode: Literal["qualified", "development"] = Field(
        default="qualified", exclude_if=lambda value: value == "qualified"
    )
    expected_release_manifest_sha256: Digest
    executor_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    models: tuple[CampaignModelV1, ...] = Field(min_length=1)
    panel: CampaignPanelV1 = Field(default_factory=CampaignPanelV1)
    operations: CampaignOperationsV1
    analysis: CampaignAnalysisV1 = Field(default_factory=CampaignAnalysisV1)
    dispatch: CampaignDispatchV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        """Reject ambiguous model identities and impossible campaign reservations."""
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError("Campaign model names must be unique")
        if len({model.model for model in self.models}) != len(self.models):
            raise ValueError("Declare one route per model per campaign; route ablations use separate conditions")
        if self.dispatch is not None:
            priorities = tuple((row.benchmark, row.task_id) for row in self.dispatch.task_order)
            expected = {
                (benchmark, task_id)
                for benchmark in self.panel.benchmarks
                for task_id in (self.panel.trialeval_tasks if benchmark == "trialeval" else self.panel.trialdev_tasks)
            }
            if any(
                not (self.panel.trialeval_tasks if benchmark == "trialeval" else self.panel.trialdev_tasks)
                for benchmark in self.panel.benchmarks
            ):
                raise ValueError("Dispatch requires explicit full task IDs for every benchmark")
            if len(set(priorities)) != len(priorities) or set(priorities) != expected:
                raise ValueError("Dispatch order must cover the complete panel exactly once")
            if self.dispatch.tasks_per_slice > self.operations.task_workers:
                raise ValueError("Dispatch tasks_per_slice must fit task_workers")
        if any(model.reservation_micro_usd > self.operations.spend_limit_micro_usd for model in self.models):
            raise ValueError("One request cannot fit the campaign spending limit")
        for route in {model.upstream_provider for model in self.models}:
            limits = {
                (model.requests_per_minute, model.tokens_per_minute)
                for model in self.models
                if model.upstream_provider == route
            }
            if len(limits) != 1:
                raise ValueError("Models sharing a provider route must declare the same shared capacity limits")
        return self


class CampaignRunV1(_Model):
    """A native scheduled run and exact immutable invocation, never a second task ledger.

    ``retry_of_job_id`` links infrastructure-failed tasks to their primary job.
    Supplementary runs retain the audit that reclassified previously missed tasks.
    """

    job_id: Identifier
    benchmark: Literal["trialeval", "trialdev"]
    model_name: Identifier
    repetition: int = Field(ge=1)
    relative_run_root: str = Field(min_length=1)
    invocation_sha256: Digest
    analysis_plan_path: str | None = None
    analysis_plan_sha256: Digest | None = None
    retry_of_job_id: Identifier | None = Field(default=None, exclude_if=lambda value: value is None)
    reclassification_audit_path: str | None = Field(default=None, exclude_if=lambda value: value is None)
    reclassification_audit_sha256: Digest | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Keep all run and plan paths below their campaign root."""
        for text in (self.relative_run_root, self.analysis_plan_path, self.reclassification_audit_path):
            if text is not None:
                path = PurePosixPath(text)
                if (
                    not path.parts
                    or path.is_absolute()
                    or ".." in path.parts
                    or "\\" in text
                    or path.as_posix() != text
                ):
                    raise ValueError("Campaign custody paths must remain normalized and relative")
        if not self.relative_run_root.startswith("runs/"):
            raise ValueError("Native invocation custody must belong to the campaign runs directory")
        if self.benchmark == "trialeval":
            if (
                self.analysis_plan_path is None
                or self.analysis_plan_sha256 is None
                or not self.analysis_plan_path.startswith("plans/")
            ):
                raise ValueError("TrialEval runs require their prospectively bound analysis plan")
        elif self.analysis_plan_path is not None or self.analysis_plan_sha256 is not None:
            raise ValueError("TrialDev runs use native invocation custody, not a TrialEval analysis plan")
        if (self.reclassification_audit_path is None) != (self.reclassification_audit_sha256 is None):
            raise ValueError("Reclassification audit path and digest must be supplied together")
        if self.reclassification_audit_path is not None and self.retry_of_job_id is None:
            raise ValueError("Only a retry run can bind a reclassification audit")
        return self


class CampaignPlanV1(_Model):
    """Complete pre-observation campaign bound to native run identities.

    Retry runs are appended only under the prespecified infrastructure-retry
    policy: at most one retry per TrialEval primary task (repetition one).
    """

    schema_id: Literal["trialagentbench.campaign_plan/v1"] = "trialagentbench.campaign_plan/v1"
    configuration: CampaignConfigurationV1
    runs: tuple[CampaignRunV1, ...] = Field(min_length=1)
    checksum: Digest | None = None

    @model_validator(mode="after")
    def bind_plan(self) -> Self:
        """Require the complete configured run inventory and content identity."""
        expected = {
            (model.name, benchmark, repetition)
            for model in self.configuration.models
            for benchmark in self.configuration.panel.benchmarks
            for repetition in range(1, self.configuration.panel.repetitions + 1)
        }
        scheduled = tuple(run for run in self.runs if run.retry_of_job_id is None)
        observed = {(run.model_name, run.benchmark, run.repetition) for run in scheduled}
        if observed != expected or len(observed) != len(scheduled):
            raise ValueError("Campaign native runs do not cover the configured inventory")
        originals = {run.job_id: run for run in scheduled}
        retried: set[str] = set()
        for run in self.runs:
            if run.retry_of_job_id is None:
                continue
            original = originals.get(run.retry_of_job_id)
            if (
                original is None
                or original.benchmark != "trialeval"
                or original.repetition != 1
                or (run.benchmark, run.model_name, run.repetition) != ("trialeval", original.model_name, 1)
            ):
                raise ValueError("Campaign retry run must repeat one TrialEval primary job of the same model")
            if run.retry_of_job_id in retried and run.reclassification_audit_path is None:
                raise ValueError("Supplementary infrastructure retry runs require a reclassification audit")
            retried.add(run.retry_of_job_id)
        if len({run.job_id for run in self.runs}) != len(self.runs):
            raise ValueError("Campaign job identities must be unique")
        if len({run.relative_run_root for run in self.runs}) != len(self.runs):
            raise ValueError("Campaign runs must have separate native custody directories")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Campaign plan checksum mismatch")
        object.__setattr__(self, "checksum", digest)
        return self


# Censoring-rate gap, in absolute proportion of scheduled tasks, above which
# paired comparisons carry an explicit informative-censoring warning.
MATERIAL_CENSORING_DIFFERENCE = 0.05


class CensoredRunV1(_Model):
    """One native job's scheduled tasks and those an exhausted spending budget left unrun."""

    job_id: Identifier
    model_name: Identifier
    model_id: str = Field(min_length=1)
    benchmark: Literal["trialeval", "trialdev"]
    repetition: int = Field(ge=1)
    scheduled_task_ids: tuple[str, ...] = Field(min_length=1)
    censored_task_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Censor only unique scheduled tasks."""
        if len(set(self.scheduled_task_ids)) != len(self.scheduled_task_ids):
            raise ValueError("Scheduled task identities must be unique")
        if len(set(self.censored_task_ids)) != len(self.censored_task_ids) or not set(self.censored_task_ids) <= set(
            self.scheduled_task_ids
        ):
            raise ValueError("Budget-censored tasks must be unique scheduled tasks")
        return self

    @property
    def wholly_censored(self) -> bool:
        """Return whether the exhausted budget left none of this job's tasks run."""
        return len(self.censored_task_ids) == len(self.scheduled_task_ids)


class ModelCensoringV1(_Model):
    """One model's budget censoring on one benchmark, across scheduled repetitions."""

    benchmark: Literal["trialeval", "trialdev"]
    model_id: str = Field(min_length=1)
    scheduled_tasks: int = Field(ge=1)
    budget_censored_tasks: int = Field(ge=0)
    censoring_rate: float = Field(ge=0, le=1)


class BudgetCensoringV1(_Model):
    """Episodes an exhausted operator spending limit can never run, separated from participant failures.

    A pending task is budget-censored only when the campaign ledger records the
    permanent exhaustion of its model's (or the campaign's) spending limit.
    Censored tasks are unscored, not missing answers. Comparisons and
    leaderboards use the paired panel of tasks no model lost to censoring;
    ``models`` and ``warnings`` are derived from ``runs`` and validated.
    """

    schema_id: Literal["trialagentbench.campaign_budget_censoring/v1"] = "trialagentbench.campaign_budget_censoring/v1"
    campaign_plan_sha256: Digest
    exhausted_models: tuple[Identifier, ...] = ()
    campaign_budget_exhausted: bool = False
    runs: tuple[CensoredRunV1, ...] = Field(min_length=1)
    models: tuple[ModelCensoringV1, ...] = ()
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def derive_summaries(self) -> Self:
        """Require recorded exhaustion for every censored run and derive per-model rates."""
        if len({run.job_id for run in self.runs}) != len(self.runs):
            raise ValueError("Censoring records repeat a campaign job")
        for run in self.runs:
            if (
                run.censored_task_ids
                and run.model_name not in self.exhausted_models
                and not self.campaign_budget_exhausted
            ):
                raise ValueError("Budget censoring requires a recorded spending-limit exhaustion")
        totals: dict[tuple[str, str], list[int]] = {}
        for run in self.runs:
            counts = totals.setdefault((run.benchmark, run.model_id), [0, 0])
            counts[0] += len(run.scheduled_task_ids)
            counts[1] += len(run.censored_task_ids)
        models = tuple(
            ModelCensoringV1(
                benchmark=benchmark,
                model_id=model_id,
                scheduled_tasks=scheduled,
                budget_censored_tasks=censored,
                censoring_rate=censored / scheduled,
            )
            for (benchmark, model_id), (scheduled, censored) in sorted(totals.items())
        )
        warnings = []
        for benchmark in sorted({row.benchmark for row in models}):
            rates = [row for row in models if row.benchmark == benchmark]
            if not self.paired_task_ids(benchmark):
                warnings.append(
                    f"{benchmark}: no scheduled task was completed by every model; "
                    "comparisons and the leaderboard are unavailable."
                )
            if max(row.censoring_rate for row in rates) - min(row.censoring_rate for row in rates) > (
                MATERIAL_CENSORING_DIFFERENCE
            ):
                listed = ", ".join(f"{row.model_id} {row.censoring_rate:.0%}" for row in rates)
                warnings.append(
                    f"{benchmark}: budget censoring differs materially between models ({listed}). "
                    "Comparisons and the leaderboard use only tasks every model completed; "
                    "they may not represent the scheduled panel."
                )
        for name, derived in (("models", models), ("warnings", tuple(warnings))):
            supplied = getattr(self, name)
            if supplied and supplied != derived:
                raise ValueError(f"Budget censoring {name} differ from their runs")
            object.__setattr__(self, name, derived)
        return self

    def censored_by_job(self, benchmark: Literal["trialeval", "trialdev"]) -> dict[str, frozenset[str]]:
        """Return each benchmark job's budget-censored task identities."""
        return {run.job_id: frozenset(run.censored_task_ids) for run in self.runs if run.benchmark == benchmark}

    def censored_count(self, benchmark: Literal["trialeval", "trialdev"], model_id: str) -> int:
        """Count one model's budget-censored tasks on one benchmark."""
        return sum(
            row.budget_censored_tasks for row in self.models if (row.benchmark, row.model_id) == (benchmark, model_id)
        )

    def paired_task_ids(self, benchmark: Literal["trialeval", "trialdev"]) -> frozenset[str]:
        """Return scheduled tasks that no model lost to budget censoring in any repetition."""
        runs = [run for run in self.runs if run.benchmark == benchmark]
        scheduled = frozenset(task for run in runs for task in run.scheduled_task_ids)
        return scheduled - {task for run in runs for task in run.censored_task_ids}
