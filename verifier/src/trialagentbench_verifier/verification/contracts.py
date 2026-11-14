"""Typed inputs for verification orchestration and claim-level reporting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.properties.contracts import FileDigest, Verdict


class Model(BaseModel):
    """Closed verification boundary with finite numerical values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ReportContext(Model):
    """Bound declarations and supplementary evidence needed to state claim scope."""

    policy: FileDigest
    payload: FileDigest
    gates: FileDigest
    profiles: FileDigest
    cell_declarations: FileDigest
    external_quantities: FileDigest
    external_source: FileDigest
    sequential_fields: FileDigest
    run_metadata: FileDigest | None = None
    deterministic_summary: FileDigest | None = None


class ReportPolicy(Model):
    """Explicit reporting allowances and field names supplied by the run manifest."""

    context_relative_tolerance: float = Field(gt=0)
    context_scale_floor: float = Field(gt=0)
    schedule_alpha: float = Field(gt=0, lt=1)
    schedule_alpha_tolerance: float = Field(gt=0)
    rescore_relative_tolerance: float = Field(gt=0)
    binding_fields: tuple[str, ...] = Field(min_length=1)
    sequential_dataset: str
    external_release: str


class ClaimNumber(Model):
    """One public number and the exact source fields and reduction producing it."""

    value: int | float | str
    sources: tuple[FileDigest, ...]
    fields: tuple[str, ...]
    operation: str
    display: str | None = None
    display_format: Literal[".6g"] | None = None


class ReportInputs(Model):
    """Authenticated layer results; private inputs are never copied to the panel."""

    schema_id: Literal["trialagentbench.verification.report_inputs/v1"] = (
        "trialagentbench.verification.report_inputs/v1"
    )
    deterministic: FileDigest
    statistical: FileDigest
    repro: FileDigest
    observation: tuple[FileDigest, ...]
    rescore: FileDigest
    pipeline_cells: FileDigest
    positive_control: FileDigest
    external_routes: FileDigest
    real_trial: FileDigest
    realism_ranges: FileDigest
    sequential: FileDigest
    nb2: FileDigest
    nb2_census: FileDigest
    smoke: FileDigest | None = None
    context: ReportContext | None = None


class PublicReport(Model):
    """Only aggregate positive claims, with counts and digests for withheld data."""

    schema_id: Literal["trialagentbench.verification.report/v1"] = "trialagentbench.verification.report/v1"
    claims: tuple[Verdict, ...]
    provenance: str = ""


class SourceProjection(BaseModel):
    """Typed fields consumed from an independently owned, authenticated receipt."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class ReproComparison(SourceProjection):
    """Stored-reference comparison emitted by the R acceptance runner."""

    field: str
    source: str
    difference: float
    tolerance: float = Field(ge=0)
    margin: float
    passed: bool = Field(alias="pass")
    cell: str
    route_id: str | None
    task_id: str
    replication: int | None


class RescoreComparison(SourceProjection):
    """Independent re-scoring comparison classification and independent diagnostics."""

    status: Literal[
        "matched_exact",
        "matched_1e-12",
        "matched_display",
        "mismatched",
        "conditional",
        "not_reproducible",
        "no_stored_target",
    ]
    stage: str
    binary64_equal: bool | None
    relative_1e12: bool | None
    display_equal: bool
    display_cell: bool


class ExternalRoutes(SourceProjection):
    """Exact retained external-reference gate counts."""

    accepted: bool
    passed_count: int = Field(ge=0)
    scheduled_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class ObservationComponent(SourceProjection):
    """Availability of a component in the standalone observation receipt."""

    parameter: str
    availability: Literal["available", "unavailable"]


class ObservationReceipt(SourceProjection):
    """Standalone model-fit receipt projection, without coefficient values."""

    components: tuple[ObservationComponent, ...]


class PositiveControl(SourceProjection):
    """Composite decision on an absent-property control."""

    cell_state: Literal["FALSIFIED", "NOT_FALSIFIED", "UNAVAILABLE"]
    cell_p: float = Field(ge=0, le=1)
    and_multiplier: int = Field(gt=0)


class RealTrial(SourceProjection):
    """Independent real-trial comparison counts and original tolerances."""

    status: str
    trials: int = Field(ge=0)
    complete_participants: int = Field(ge=0)
    numeric_comparisons: int = Field(ge=0)
    passed_comparisons: int = Field(ge=0)
    r_fit_failures: int = Field(ge=0)
    maximum_tolerance_fraction: float
    rtol: float
    atol: float


class RangeComparison(SourceProjection):
    """Classification only; mechanism values have no report representation."""

    private: bool
    inside_outside: str
    classification: str
    cell: str


class SequentialSplit(SourceProjection):
    """Claim-level result of the declared-precision sequential comparison."""

    split: Literal["public", "withheld"]
    passed: bool


class SequentialReceipt(SourceProjection):
    """Public and withheld sequential checks, without per-field values."""

    results: tuple[SequentialSplit, ...]


class NB2Receipt(SourceProjection):
    """Identifiable stored-NB2 fit comparison census."""

    compared: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    counts: dict[str, int]


class NB2CensusRow(Model):
    """Complete field-level census schema, including unsuccessful and unbound fits."""

    readout_id: str
    readout_root: str
    arm_id: str | None
    release_id: str | None
    pool_manifest_sha256: str | None
    binding_evidence: str | None
    n: int | None
    events: int | None
    field: Literal["estimate", "standard_error", "interval_lower", "interval_upper"] | None
    r: float | None
    r_at_declared_precision: float | None
    stored: float | None
    difference: float | None
    tolerance: float | None = Field(ge=0)
    status: Literal["unbindable", "unavailable", "crashed", "passed", "failed", "non-converged"]
    stored_source: str | None
    stored_method_evidence: str | None
    message: str | None


class SmokeReceipt(SourceProjection):
    """Optional completed smoke receipt; unavailable work adds no public claim."""

    status: str
    source_commit: str
    executed_comparisons: int = Field(ge=0)


class U7TestFile(FileDigest):
    """A pinned test input and its portable container-relative binding."""

    target: str


class RTestEnvironment(Model):
    """Required bindings for the merged package's real-input acceptance tests."""

    TABVERIFY_INPUTS: str
    TABVERIFY_DECLARATIONS: str
    TABVERIFY_RECEIPT: str
    TABVERIFY_BANK: str
    TABVERIFY_REALISM_DECLARATION: str
    TABVERIFY_REALISM_SHA256: str


class U7Paths(Model):
    """Physical and historical logical bindings supplied by the run manifest."""

    data_root: str
    source_prefix: str
    declaration: FileDigest
    keys: FileDigest
    paper_table: FileDigest
    prior_comparison: FileDigest
    test_files: tuple[U7TestFile, ...]
    test_environment: RTestEnvironment
    replay_scan: str
    dependency_library: str
    dependency_sources: str
    forbidden_source_roots: tuple[str, ...]


class VerificationInputs(Model):
    """Complete opt-in run configuration, resolving paths from this manifest."""

    schema_id: Literal["trialagentbench.verification.run_inputs/v1"] = "trialagentbench.verification.run_inputs/v1"
    release_relative: str
    anchor: FileDigest
    deterministic_inputs: FileDigest
    deterministic_declarations: str
    statistical_inputs: FileDigest
    r_dependencies: str
    repro_inputs: str
    repro_files: tuple[FileDigest, ...]
    observation_index: FileDigest
    observation_declaration: FileDigest
    observation_files: tuple[FileDigest, ...]
    u7: U7Paths
    receipts: ReportInputs
    s4_pins: tuple[FileDigest, ...]
    statistical_public: tuple[FileDigest, ...]
