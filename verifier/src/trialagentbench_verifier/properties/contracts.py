"""Data contracts for independent trial-property verification.

R request-directory protocol
----------------------------
Each fit receives its own directory containing ``request.json`` (``RPayload``)
and the CSV files named by ``tables``. Paths in the request are relative to that
directory. The caller creates a separate output directory and invokes::

    docker run --rm --network none -v req:/in:ro -v out:/out <image> \
        Rscript -e 'tabverify::run("/in","/out")'

``req`` and ``out`` are absolute host directories. The offline backend
authenticates its image and dependency versions before executing R.
R runs with one numerical thread and no randomness. It writes
``/out/response.json`` (the ``RResult`` discriminated union) and
``/out/sessionInfo.txt``. The ``tabverify`` request protocol owns the
wire schema: estimates select a declared cell/route pair or an explicit panel
configuration. Parameter and coefficient routines share the ``mechanism``
response envelope, retaining pooled counts, exposure and component availability.
The legacy ``mechanism`` request is descriptive; declaration tests use
``mechanism_parameters``. ``pool_truth`` remains a reserved extension point.
Validate and serialize both wire unions with ``pydantic.TypeAdapter``.

Serialization and scope
-----------------------
All records forbid unknown fields. JSON schemas carry a versioned ``$id``;
artifact envelopes also carry ``schema_id``; R wire messages do not. Use ``exclude_unset=True`` when
serializing declarations to retain the supplied declaration's JSON structure,
including omitted variant fields. This is JSON-value preservation, not whitespace
preservation. Declarations may omit process profiles and release precision:
profiles can be supplied separately and precision defaults to the declared 6.

Only paths, identities and contracts live here; no files are opened or checks
executed. ``materials``, ``key_record`` and failure evidence retain authenticated
JSON whose schemas belong to their producers. They are not interpreted here,
except for the declared task-key binding to ``key_record.target.value``.
The public export and attestation contain positive claim rows. Unavailable
checks, stopped-cell details and residual findings belong in internal detail.
Withheld claim evidence contains counts and digests only.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    StringConstraints,
    model_serializer,
    model_validator,
)

Text = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
SourceCommit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Probability = Annotated[float, Field(ge=0, le=1)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]
Split = Literal["public", "withheld"]
AssumptionTier = Literal["A1", "A2", "A3"]
Context = Literal["C1", "C2", "C3", "C4", "C5"]
DesignTier = Literal["D1", "D2", "D3", "D4"]
Status = Literal["pass", "fail", "unavailable"]
CheckId = Literal[
    "L1-ID",
    "L1-SPLIT",
    "L1-COUNT",
    "L1-CTX",
    "L1-DESIGN",
    "L1-BIND",
    "L1-SUPPORT",
    "L1-DESIGNTIME",
    "L1-DIAG",
    "L1-REPRO",
    "L1-MECH",
    "L1-OBS",
    "L1-ICE",
    "L1-CENS",
    "L1-CAL",
    "L1-REC",
    "L1-CONSEQ-A3",
    "L1-REGRET-A2",
]
Family = Literal["F-MECH", "F-OBS", "F-ICE", "F-CENS", "F-CAL", "F-REC", "F-CONSEQ", "F-REGRET"]
_CELL_NON_NULL_MECHANISM_FIELDS = frozenset(
    {
        "calendar_final_log_hazard_offset",
        "censoring_scale_multiplier",
        "controlled_condition",
        "endpoint_validation_strata",
        "intercurrent_events",
    }
)
_CELL_MECHANISM_FIELDS = _CELL_NON_NULL_MECHANISM_FIELDS | {"count_dispersion", "identified_set", "observation"}


def _schema_identity(schema: dict[str, Any], model: type[BaseModel]) -> None:
    name = re.sub(r"(?<!^)(?=[A-Z][a-z]|(?<=[a-z0-9])[A-Z])", "_", model.__name__).lower()
    if model.__name__ == "RunInputManifest":
        name = "run_inputs"
    schema["$id"] = f"trialagentbench.properties.{name}/v1"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra=_schema_identity)


class CrosswalkRow(_Model):
    """One released cell reference and its declared regime identity."""

    cell_ref: Text
    regime_cell_id: Text


class PrivateTaskAssignment(_Model):
    """Private construction-catalog assignment, never a public export row."""

    task_id: Text
    context: Context
    task_key: Text
    scenario_label: Text
    declared_class: Text


class DeclarationRegisterRow(_Model):
    """One field-to-check CSV row, preserving its JSON-encoded declared value."""

    source: Text
    scope: Text
    field: Text
    declared: str
    check: CheckId | Literal["none", "L2-REALISM", "L1-REC+L1-REPRO", "L1-CONSEQ-A3/L1-REGRET-A2"]
    rule: Text


class AssumptionDefinition(_Model):
    """Declared A-tier obligations, retained without interpreting their gates."""

    tier: AssumptionTier
    name: Text
    corrected_route_must_pass: bool
    default_route_must_fail: bool
    default_route_must_pass: bool
    permitted_public_evidence_kinds: tuple[str, ...]
    practical_consequence_required: bool


class ContextDefinition(_Model):
    """Declared evidence flags for one C-tier."""

    tier: Context
    analysis_ready_tables: bool
    controlled_recoverable_defect: bool
    locked_sap: bool
    raw_participant_evidence: bool
    reconstruction_required: bool


class DesignDefinition(_Model):
    """Declared allocation, method features and evidence for one D-tier."""

    tier: DesignTier
    allocation_structure: Text
    method_changing_features: tuple[str, ...]
    name: Text
    required_public_evidence: tuple[str, ...]


class DesignProfile(_Model):
    """Design-profile identities and the series to which they apply."""

    allocation_unit: Text
    analysis_defining_subdesign: Text
    analysis_design_family: Text
    design_profile_id: Text
    design_tier: DesignTier
    evaluation_series_ids: tuple[str, ...]
    intervention_model: Text
    study_format: Text


class _Procedure(_Model):
    analysis_unit: Text
    arm_quantities: tuple[str, ...]
    combination: Text
    confidence_level: Probability
    contrast_quantity: Text
    degrees_of_freedom: Text
    fitting_specification: Text
    grouping: Text
    partition: Text
    refitting: Text
    transformation: Text
    variance: Text


class Procedure(_Procedure):
    """Normative fitting and uncertainty instructions for an analysis route."""

    resampling_count: PositiveInt | Literal["all_independent_units"] | None
    resampling_rule: str | None
    resampling_seed: int | None
    resampling_unit: str | None


class ReferenceAnalysis(_Model):
    """One credited analysis or excluded negative-control analysis."""

    design_obligations: tuple[str, ...]
    diagnostic_requirements: tuple[str, ...]
    effect_scale: Text
    estimator_family: Text
    implementation_summary: Text
    interpretation: Text
    label: Text
    method_family_id: str | None
    procedure: Procedure
    result_kind: Text
    route_id: Text
    uncertainty_method: Text


class ArmAccuracy(_Model):
    """Declared binary recording accuracy for one randomized arm."""

    arm_id: Text
    sensitivity: Probability
    specificity: Probability


class ArmCapture(_Model):
    """Declared count-event capture probability for one randomized arm."""

    arm_id: Text
    capture_probability: Probability


class BinaryValidationStratum(_Model):
    """Binary endpoint validation, with nullable shared-accuracy fields."""

    value: Text
    validation_probability: Probability
    accuracy_by_arm: tuple[ArmAccuracy, ...]
    sensitivity: Probability | None
    specificity: Probability | None


class CountValidationStratum(_Model):
    """Count endpoint validation with arm-specific capture probabilities."""

    value: Text
    validation_probability: Probability
    capture_by_arm: tuple[ArmCapture, ...]


class MainEffect(_Model):
    """Declared scalar effect of one input variable."""

    effect_type: Literal["main"]
    effect_id: Text
    input_variable: Text
    coefficient: float
    unit: Text


class InteractionEffect(_Model):
    """Declared interaction of the named input variables."""

    effect_type: Literal["interaction"]
    effect_id: Text
    input_variables: tuple[str, ...]
    coefficient: float
    unit: Text


class SplineEffect(_Model):
    """Declared spline values, knots and extrapolation rule."""

    effect_type: Literal["spline"]
    effect_id: Text
    input_variable: Text
    knots: tuple[float, ...]
    values: tuple[float, ...]
    extrapolation: Text
    unit: Text


Effect = Annotated[MainEffect | InteractionEffect | SplineEffect, Field(discriminator="effect_type")]


class ArmCoefficient(_Model):
    """A process coefficient indexed by randomized arm."""

    arm_id: Text
    value: float


class Observation(_Model):
    """Declared observation process; inactive, undeclared extensions stay empty."""

    baseline_coefficient: float
    effects: tuple[Effect, ...]
    event_coefficients: tuple[()]
    hierarchical_effect_correlations: tuple[()]
    initial_observed_outcome: float
    intercept: float
    learned_effect: None
    monotone_dropout: bool
    observe_first_contact: bool
    process_type: Text
    state_coefficients: tuple[()]
    terminal_missingness_only: bool
    terminal_recovery_probability: Probability
    time_coefficient: float


class IntercurrentEvent(_Model):
    """Declared contact-level event model and post-event exposure action."""

    arm_coefficients: tuple[ArmCoefficient, ...]
    baseline_coefficient: float
    baseline_variable: Text
    cluster_probability_concentration: float | None
    cluster_reference_probability: Probability | None
    effects: tuple[Effect, ...]
    event_id: Text
    event_type: Text
    exposure_action: Text
    hierarchical_effect_correlations: tuple[()]
    intercept: float
    kind: Text
    learned_effect: None
    occurrence_timing: Text
    outcome_shift: float
    recurrent: bool
    risk_reference_duration: float | None
    risk_window_end: float | None
    risk_window_start: float
    state_coefficients: tuple[()]
    target_arm: str | None
    time_coefficient: float
    withdrawal_reason: str | None
    withdrawal_reason_text: str | None


class IdentifiedSet(_Model):
    """Descriptive identified-set declaration, without a verification gate."""

    functional: Text
    identification_assumption_ids: tuple[str, ...]
    sensitivity_parameters: tuple[()]
    set_definition_id: Text


class CellDeclaration(_Model):
    """One cell with full mechanism declarations or tier-only scope.

    Notes
    -----
    Mechanism fields are omitted in tier-only cells, including fields whose
    full declaration can be null. Serializing with ``exclude_unset=True``
    preserves that distinction. Full cells retain every declared field.
    """

    assumption_tier: AssumptionTier
    declaration_scope: Literal["full", "tier_only"]
    calendar_final_log_hazard_offset: float | None = None
    censoring_scale_multiplier: Annotated[float, Field(gt=0)] | None = None
    context_ids: tuple[Context, ...]
    controlled_condition: Text | None = None
    count_dispersion: float | None = None
    default_response: Literal["compatible", "tempting_correction_fails", "material_failure_with_alternative"]
    endpoint_validation_strata: tuple[BinaryValidationStratum | CountValidationStratum, ...] | None = None
    evaluation_series_id: Text
    identification_class: Text
    identified_set: IdentifiedSet | None = None
    intercurrent_events: tuple[IntercurrentEvent, ...] | None = None
    negative_control: ReferenceAnalysis | None
    observation: Observation | None = None
    observation_kind: str | None
    reference_analyses: tuple[ReferenceAnalysis, ...]
    regime_cell_id: Text
    replicates_per_regime_context: PositiveInt
    withheld_only: bool = False

    @model_validator(mode="after")
    def validate_declaration_scope(self) -> Self:
        """Require full declarations and reject mechanism fields in tier-only cells."""
        if self.declaration_scope == "full":
            missing = _CELL_MECHANISM_FIELDS - self.model_fields_set
            if missing:
                raise ValueError(f"full cells require declared mechanism fields: {sorted(missing)}")
            if any(getattr(self, name) is None for name in _CELL_NON_NULL_MECHANISM_FIELDS):
                raise ValueError("full cells require non-null values for nonnullable mechanism declarations")
        elif present := _CELL_MECHANISM_FIELDS & self.model_fields_set:
            raise ValueError(f"tier_only cells must omit mechanism fields: {sorted(present)}")
        return self

    @model_serializer(mode="wrap")
    def serialize_declaration_scope(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit tier-only mechanism fields even with default serialization options."""
        payload: dict[str, Any] = handler(self)
        if self.declaration_scope == "tier_only":
            for name in _CELL_MECHANISM_FIELDS:
                payload.pop(name, None)
        return payload


class Distribution(_Model):
    """Declared moments and bounds of a clinical marginal distribution."""

    family: Text
    lower: float
    mean: float
    standard_deviation: Annotated[float, Field(ge=0)]
    upper: float | None


class BernoulliDistribution(_Model):
    """Analytic Bernoulli distribution."""

    family: Literal["bernoulli"]
    probability: Probability


class BetaDistribution(_Model):
    """Analytic beta distribution."""

    family: Literal["beta"]
    alpha: Annotated[float, Field(gt=0)]
    beta: Annotated[float, Field(gt=0)]


class LogNormalDistribution(_Model):
    """Analytic log-normal distribution on the declared log scale."""

    family: Literal["log_normal"]
    log_mean: float
    log_standard_deviation: Annotated[float, Field(ge=0)]


class UniformDistribution(_Model):
    """Analytic bounded uniform distribution."""

    family: Literal["uniform"]
    lower: float
    upper: float


class TruncatedNormalDistribution(_Model):
    """Analytic truncated normal distribution."""

    family: Literal["truncated_normal"]
    lower: float
    upper: float
    mean: float
    standard_deviation: Annotated[float, Field(gt=0)]


class MixtureDistribution(_Model):
    """Declared age mixture, with component ordering preserved."""

    family: Literal["mixture"]
    components: tuple[UniformDistribution | TruncatedNormalDistribution, ...]
    weights: tuple[Probability, ...]


class AnalyticQuantity(_Model):
    """Numeric clinical quantity defined by an analytic distribution."""

    quantity_type: Literal["analytic_numeric_distribution"]
    unit: Text
    distribution: Annotated[
        BernoulliDistribution | BetaDistribution | LogNormalDistribution | MixtureDistribution,
        Field(discriminator="family"),
    ]


class MomentQuantity(_Model):
    """Numeric clinical quantity defined by moments and optional bounds."""

    quantity_type: Literal["moment_target"]
    unit: Text
    family: Text
    mean: float
    variance: Annotated[float, Field(ge=0)]
    lower: float | None
    upper: float | None
    trials: int | None
    absolute_tolerance: Annotated[float, Field(ge=0)]


class DeferredQuantity(_Model):
    """Clinical quantity resolved by a named baseline mechanism."""

    quantity_type: Literal["deferred"]
    unit: Text
    reason: Text
    resolution_stage: Text


Quantity = Annotated[AnalyticQuantity | MomentQuantity | DeferredQuantity, Field(discriminator="quantity_type")]


class BaselineFeature(_Model):
    """Additional clinical baseline feature and its quantity declaration."""

    encoding: Text
    quantity: Quantity
    role: Text
    variable_id: Text


class DerivedFeatureTerm(_Model):
    """One declared power term, retaining symbolic coefficients as data."""

    effect_id: Text
    power: NonnegativeInt
    coefficient: float | Text
    standardization: Text


class DerivedFeature(_Model):
    """A derived variable and its declared inputs and transformation."""

    variable_id: Text
    mechanism_id: Text
    inputs: tuple[Text, ...] = Field(min_length=1)
    intercept: float
    terms: tuple[DerivedFeatureTerm, ...] = Field(min_length=1)
    stochastic: bool
    unit: Text
    note: Text


class DerivedFeatures(_Model):
    """Standalone derived-feature declarations, without evaluating expressions."""

    schema_id: Literal["trialagentbench.properties.derived_features/v1"] = (
        "trialagentbench.properties.derived_features/v1"
    )
    source: Text
    features: tuple[DerivedFeature, ...]


class BaselineDependence(_Model):
    """Ordered baseline dependence target and its declaration provenance."""

    attainment_tolerance: Annotated[float, Field(ge=0)]
    dependence_id: Text
    dependence_type: Text
    meaning: Text
    metric: Text
    reference_population: Text
    scale: Text
    target_matrix: tuple[tuple[float, ...], ...]
    unit: Text
    variables: tuple[str, ...]


class MeasurementError(_Model):
    """Measurement error declaration present in the supplied release."""

    error_model: Literal["exact"]


class Measurement(_Model):
    """Clinical field recording method, resolution and missing-value encoding."""

    artifact_modes: tuple[str, ...]
    device_or_assay: str | None
    error: MeasurementError
    method: Text
    missing_value: str
    resolution: float | None


class Support(_Model):
    """Clinical field support, with an explicitly nullable upper bound."""

    lower: float
    upper: float | None


class Timing(_Model):
    """Clinical field timing relative to its declared reference."""

    reference: Text
    repeatable: bool
    window_unit: str | None


class ClinicalField(_Model):
    """Baseline or primary-endpoint field, support and recording contract."""

    allowed_values: tuple[float | str, ...] | None
    categories: tuple[str, ...]
    evidence_role: Text
    family: Text
    field_id: Text
    label: Text
    measurement: Measurement
    offset: float
    role: Text
    scale: float
    settings: tuple[str, ...]
    source_codes: tuple[float, ...]
    source_direction: Literal[-1, 1]
    source_field_id: Text
    source_ids: tuple[str, ...]
    support: Support | None
    timing: Timing
    unit: Text
    value_type: Literal["categorical", "continuous", "integer"]


class MechanismParameter(_Model):
    """One fixed parameter of a declared baseline mechanism."""

    input_variable: str | None
    parameter_id: Text
    role: Text
    source: Text
    unit: Text
    value: float


class BaselineMechanism(_Model):
    """Declared baseline transformation, retaining inactive null fields."""

    effects: tuple[Effect, ...]
    hierarchical_effect_correlations: tuple[()]
    inputs: tuple[str, ...]
    learned: None
    measurement_model: None
    mechanism_id: Text
    mechanism_type: Text
    nominal_calibration: None
    nominal_logits: None
    observation_rule: None
    output: Text
    output_levels: None
    output_unit: Text
    parameters: tuple[MechanismParameter, ...]
    probability_bounds: None
    residual_scale: float | None
    stochastic: bool


class Recruitment(_Model):
    """Declared recruitment sites, countries and accrual period."""

    accrual_days: Annotated[float, Field(ge=0)]
    calendar_origin: Text
    country_codes: tuple[str, ...]
    site_count: PositiveInt | Literal["randomization_units"]
    source_id: Text


class SafetySeverity(_Model):
    """Declared safety severity category and probability."""

    level: Text
    probability: Probability
    serious: bool
    seriousness_criteria: tuple[str, ...]


class ClinicalProfile(_Model):
    """Complete clinical population, support and endpoint description."""

    additional_baseline_features: tuple[BaselineFeature, ...]
    age_distribution: Distribution
    age_prognostic_spearman: float
    age_quantity: Quantity | None
    baseline_dependence: tuple[BaselineDependence, ...]
    baseline_dependence_variables: tuple[str, ...]
    baseline_fields: tuple[ClinicalField, ...]
    baseline_mechanisms: tuple[BaselineMechanism, ...]
    baseline_spearman_matrix: tuple[tuple[float, ...], ...]
    clinical_setting: Text
    comparator_label: Text
    endpoint_definition: Text
    endpoint_event_term: Text
    endpoint_label: Text
    endpoint_object: Text
    intervention_label: Text
    joint_evidence_role: Text
    joint_source_ids: tuple[str, ...]
    population: Text
    prognostic_direction: Literal[-1, 1]
    prognostic_distribution: Distribution
    recruitment: Recruitment
    safety_event_category: Text
    safety_event_definition: Text
    safety_event_label: Text
    safety_event_risk_window_end_days: float | None
    safety_event_term: Text
    safety_reference_risk_by_horizon: Probability
    safety_severity_levels: tuple[SafetySeverity, ...]
    secondary_direction: Literal[-1, 1]
    secondary_distribution: Distribution
    study_name: Text
    therapeutic_area: Text
    time_origin: Text
    treatment_end_days: float | None


class IntercurrentStrategy(_Model):
    """Estimand handling of a named intercurrent event."""

    composite_value: float | None
    event_id: Text
    principal_stratum: str | None
    strategy: Text


class EndpointProcess(_Model):
    """Declared continuous or longitudinal endpoint process."""

    age_effect_per_standard_deviation: float
    change_point_slope_change_per_day: float | None
    change_point_time_days: float | None
    intercept: float
    prognostic_effect_per_standard_deviation: float
    progressive_state_outcome_shift: float | None
    reference_treatment_effect_magnitude: float
    residual_standard_deviation: Annotated[float, Field(ge=0)]
    secondary_effect_per_standard_deviation: float
    temporal_structure: Text
    time_coefficient_per_day: float


class BinaryEndpointProcess(_Model):
    """Declared binary endpoint reference risk and effect magnitude."""

    process_kind: Literal["binary"]
    control_probability: Probability
    reference_risk_difference_magnitude: float


class CountEndpointProcess(_Model):
    """Declared count endpoint reference mean, dispersion and rate ratio."""

    process_kind: Literal["count"]
    control_mean: Annotated[float, Field(ge=0)]
    dispersion: Annotated[float, Field(gt=0)]
    reference_favorable_rate_ratio: Annotated[float, Field(gt=0)]


class SeriesDeclaration(_Model):
    """Complete series declaration, including clinical and endpoint processes."""

    accrual_duration_days: float | None
    analysis_period_days: float
    assessment_horizon_days: float
    assumption_tiers: tuple[AssumptionTier, ...]
    available_effect_scales: tuple[str, ...]
    censoring_handling_id: Text
    clinical_profile: ClinicalProfile
    competing_event_handling_id: Text
    contact_times_days: tuple[float, ...]
    contrast_direction: Text
    default_route_id: Text
    design_profile_id: Text
    endpoint: Text
    endpoint_confirmation_delay_days: float
    endpoint_family: Text
    estimand_id: Text
    evaluation_series_id: Text
    favorable_direction: Literal["higher", "lower"]
    follow_up_loss_hazard_per_year: float | None
    intercurrent_event_strategies: tuple[IntercurrentStrategy, ...]
    label: Text
    longitudinal_time_model: str | None
    missing_observation_handling_id: Text
    monitoring_times_days: tuple[float, ...]
    primary_discrete_endpoint_process: (
        Annotated[BinaryEndpointProcess | CountEndpointProcess, Field(discriminator="process_kind")] | None
    )
    primary_effect_scale: Text
    primary_endpoint_field: ClinicalField | None
    primary_endpoint_id: Text
    primary_endpoint_process: EndpointProcess | None
    primary_estimand_id: Text
    process_transition_times_days: tuple[float, ...]
    public_question: Text
    reference_population_profile: Text
    result_unit: Text
    safety_handling_id: Text
    secondary_endpoint_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    target_population: Text
    treatment_effect_time_profile: tuple[float, ...] | None


class RecoveryCriteria(_Model):
    """Construction recovery margins, not verifier-selected thresholds."""

    bias_margin_sd: float
    confidence_level: Probability
    coverage_alpha: Probability
    coverage_lower_bound: Probability
    coverage_worlds_per_cell: PositiveInt
    minimum_worlds_per_cell: PositiveInt
    support_lower_bound: Probability
    tertile_bias_margin_sd: float


class Acquisition(_Model):
    """Declared construction-world counts, retained descriptively."""

    TRIALEVAL_COVERAGE_WORLDS_PER_CELL: PositiveInt
    TRIALEVAL_RECOVERY_WORLDS_PER_CELL: PositiveInt


class Consequence(_Model):
    """Construction consequence constants with their original field names."""

    TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD: float
    TRIALEVAL_CONSEQUENTIAL_SHIFT_BOUND_SD: float
    TRIALEVAL_CONSEQUENTIAL_SHIFT_SD: float
    TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO: float
    TRIALEVAL_NO_INVERSION_CONFIDENCE: Probability


class EconomicsPoint(_Model):
    """One value and cost multiplier pair in the declared economics band."""

    cost: Annotated[float, Field(gt=0)]
    value: Annotated[float, Field(gt=0)]


class Gates(_Model):
    """Complete exported recovery, consequence and acquisition declarations."""

    acquisition: Acquisition
    consequence: Consequence
    economics_band: tuple[EconomicsPoint, ...]
    over_correction_margin: float
    recovery_criteria: RecoveryCriteria


class SeverityMetricNames(_Model):
    """Registry metric names for each descriptive severity band."""

    broken: Text
    fragile: Text
    stressed: Text


class SeverityThresholds(_Model):
    """Registry severity definitions, retained without imposing a tier gate."""

    broken: float
    confidence_level: Probability
    decision_metric_names: SeverityMetricNames
    fragile: float
    metric_name: Text
    metric_unit: Text
    multiplicity_adjustment: Text
    stressed: float


class DiagnosticEntry(_Model):
    """Registry procedure and reporting instruction for one diagnostic key."""

    diagnostic_scale: Text
    interpretation: Text
    metric_units: dict[str, str]
    nonidentified_evidence_class: str | None
    primary_credit_policy: Text
    primary_method_change_threshold: float | None
    public_key: Text
    severity_thresholds: SeverityThresholds | None
    threshold: Text


class OperatingDefinition(_Model):
    """Full registry operating definition for an assumption tier."""

    label: Text
    operative_definition: Text
    participant_evidence_requirement: Text
    release_evidence_requirement: Text
    scoring_constraint: Text


class DiagnosticRegistry(_Model):
    """Standalone registry envelope, preserving its upstream schema identity."""

    schema_id: Literal["ariadne.diagnostic_registry/v1"]
    version: Text
    assumption_tier_operating_definitions: dict[AssumptionTier, OperatingDefinition]
    diagnostic_keys: dict[str, DiagnosticEntry]


class ProcessProfile(_Model):
    """One per-series or default exported clinical process profile."""

    active_event_log_odds_ratio: float
    active_outcome_effect: float
    allocation_block_size: PositiveInt
    censoring_hazard_multiplier: float
    cluster_censoring_log_hazard_standard_deviation: float
    cluster_log_hazard_standard_deviation: float
    cluster_size: PositiveInt
    cluster_size_coefficient_of_variation: float
    contact_attendance_baseline_coefficient: float
    contact_attendance_intercept: float
    contact_attendance_time_coefficient_per_day: float
    contact_availability_delay_days: float
    contact_documentation_delay_days: float
    contact_observation_delay_days: float
    contact_times_days: tuple[float, ...]
    endpoint_observation_baseline_coefficient: float
    endpoint_observation_intercept: float
    endpoint_observation_time_coefficient_per_day: float
    event_baseline_coefficient: float
    group_sequential_accrual_days: float | None
    group_sequential_information_times_days: tuple[float, ...]
    horizon_days: float
    interim_information_fractions: tuple[Probability, ...]
    intracluster_correlation: Probability
    outcome_autocorrelation: float
    outcome_baseline_coefficient: float
    outcome_intercept: float
    outcome_residual_standard_deviation: float
    outcome_time_coefficient_per_day: float
    primary_baseline_log_hazard_ratio: float
    primary_weibull_shape: float
    proportional_active_log_hazard_ratio: float
    safety_reporting_probability: Probability
    safety_resolution_probability_per_contact: Probability
    site_case_mix_standard_deviation: float
    site_outcome_standard_deviation: float
    standard_active_log_hazard_ratio: float
    stepped_wedge_accrual_days: float
    stepped_wedge_cluster_size: PositiveInt
    stepped_wedge_period_days: float
    switch_times_days: tuple[float, ...]
    time_origin: Text
    treatment_effect_multiplier: float


class ProcessProfiles(_Model):
    """Standalone process-profile export with its declared release precision."""

    te_s01: ProcessProfile = Field(alias="TE-S01")
    te_s02: ProcessProfile = Field(alias="TE-S02")
    te_s03: ProcessProfile = Field(alias="TE-S03")
    te_s04: ProcessProfile = Field(alias="TE-S04")
    te_s05: ProcessProfile = Field(alias="TE-S05")
    te_s06: ProcessProfile = Field(alias="TE-S06")
    te_s07: ProcessProfile = Field(alias="TE-S07")
    te_s08: ProcessProfile = Field(alias="TE-S08")
    te_s09: ProcessProfile = Field(alias="TE-S09")
    te_s10: ProcessProfile = Field(alias="TE-S10")
    te_s11: ProcessProfile = Field(alias="TE-S11")
    te_s12: ProcessProfile = Field(alias="TE-S12")
    te_s13: ProcessProfile = Field(alias="TE-S13")
    te_s14: ProcessProfile = Field(alias="TE-S14")
    te_s15: ProcessProfile = Field(alias="TE-S15")
    te_s16: ProcessProfile = Field(alias="TE-S16")
    te_s17: ProcessProfile = Field(alias="TE-S17")
    default: ProcessProfile
    release_numeric_decimal_places: NonnegativeInt


class CellDeclarations(_Model):
    """Data-only public or withheld declaration inventory.

    Notes
    -----
    A public declaration can omit profiles, precision and derived features. An inventory can
    embed the per-series/default map from ``process_profiles.json``, with that
    file's ``release_numeric_decimal_places`` moved to the sibling field here.
    Omitting profiles means they have not been supplied; it does not synthesize
    process declarations. Withheld-only metadata uses the same cell schema.
    """

    schema_id: Literal["trialagentbench.properties.cell_declarations/v1"] = (
        "trialagentbench.properties.cell_declarations/v1"
    )
    source_commit: SourceCommit
    split: Split
    tier_only_reason: Text
    assumption_definitions: tuple[AssumptionDefinition, ...]
    context_definitions: tuple[ContextDefinition, ...]
    design_definitions: tuple[DesignDefinition, ...]
    crosswalk: tuple[CrosswalkRow, ...]
    design_profiles: tuple[DesignProfile, ...]
    series: tuple[SeriesDeclaration, ...]
    cells: tuple[CellDeclaration, ...]
    gates: Gates
    diagnostic_registry: dict[str, DiagnosticEntry]
    process_profiles: dict[str, ProcessProfile] = Field(default_factory=dict)
    release_numeric_decimal_places: NonnegativeInt = 6
    derived_features: DerivedFeatures | None = None


class FileDigest(_Model):
    """An input path and SHA-256 of its exact bytes."""

    path: Text
    sha256: Sha256


class OutputLinks(_Model):
    """Retained output link destinations, recorded without following them."""

    schema_id: Literal["trialagentbench.properties.output_links/v1"] = "trialagentbench.properties.output_links/v1"
    links: dict[Text, Text]


class SequentialPlan(_Model):
    """Computational fields of the separately authenticated S09 export."""

    interim_information_fractions: tuple[Probability, ...]
    monitoring_times_days: tuple[float, ...]


class RecordDigest(FileDigest):
    """A private retained record, with its producer-declared schema identity."""

    schema_id: Text


class WithheldRelease(_Model):
    """Private release root bound by an anchor, root sums, or both."""

    root: Text
    anchor_sha256: Sha256 | None = None
    sums_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def require_identity(self) -> Self:
        """Require at least one of the design's two release identities."""
        if self.anchor_sha256 is None and self.sums_sha256 is None:
            raise ValueError("withheld release requires an anchor or sums digest")
        return self


class RunInputManifest(_Model):
    """Authenticated run inputs and declaration bindings.

    Notes
    -----
    Tier tables, declarations and records bind their exact bytes; an empty record list makes no availability claim.
    """

    schema_id: Literal["trialagentbench.properties.run_inputs/v1"] = "trialagentbench.properties.run_inputs/v1"
    release_id: Text
    split: Split
    declarations: FileDigest
    derived_features: FileDigest
    crosswalk: FileDigest
    tier_table: FileDigest
    declaration_files: tuple[FileDigest, ...] = ()
    task_assignments: FileDigest | None = None
    s0_task_assignment: FileDigest | None = None
    s0_flag_reference: FileDigest | None = None
    withheld_release: WithheldRelease | None = None
    trialdev_records: tuple[RecordDigest, ...]
    gate_records: tuple[RecordDigest, ...] | None = None


class VerifiedRelease(_Model):
    """Authenticated release identities handed to property checks."""

    schema_id: Literal["trialagentbench.properties.verified_release/v1"] = (
        "trialagentbench.properties.verified_release/v1"
    )
    root: Text
    anchor_sha256: Sha256
    sums_sha256: Sha256
    release_id: Text
    split: Split


class TaskVariant(_Model):
    """One context view of a task, retaining its authenticated material JSON."""

    schema_id: Literal["trialagentbench.properties.task_variant/v1"] = "trialagentbench.properties.task_variant/v1"
    task_id: Text
    cell_ref: Text
    task_key: Text
    context: Context
    materials: dict[str, JsonValue]
    key_record: dict[str, JsonValue]

    @model_validator(mode="after")
    def bind_task_key(self) -> Self:
        """Require the design's literal cell and target-value task key."""
        target = self.key_record.get("target")
        if not isinstance(target, dict) or "value" not in target:
            raise ValueError("key_record requires target.value for task_key")
        value = target["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("target.value must be numeric")
        if self.task_key != f"{self.cell_ref}#{value}":
            raise ValueError("task_key must equal cell_ref + '#' + str(target.value)")
        return self


class Replication(_Model):
    """A scored or replay dataset with paths to its released tables."""

    schema_id: Literal["trialagentbench.properties.replication/v1"] = "trialagentbench.properties.replication/v1"
    variant: TaskVariant
    kind: Literal["replay", "scored"]
    index: NonnegativeInt
    table_paths: dict[Text, Text]


EmptyObject = Annotated[dict[str, None], Field(max_length=0)]


class RProcedure(_Procedure):
    """Declared procedure retaining jsonlite's empty-object null placeholders."""

    resampling_count: PositiveInt | Literal["all_independent_units"] | EmptyObject | None
    resampling_rule: str | EmptyObject | None
    resampling_seed: int | EmptyObject | None
    resampling_unit: str | EmptyObject | None


class RConfiguration(_Model):
    """Explicit panel configuration, including the exported normative procedure."""

    kind: Text
    horizon: float | None = None
    covariates: tuple[Text, ...] = ()
    endpoint: Text | None = None
    endpoint_family: Text | None = None
    time_model: Text | EmptyObject | None = None
    basis: Text | None = None
    procedure: RProcedure | None = None
    analysis_set: Text | None = None
    bmi_field: Text | None = None
    bmi_offset: float | None = None
    bmi_scale: float | None = None
    exposure_strategy: Text | None = None
    period_length: float | None = None
    information_fractions: tuple[Probability, ...] | None = None
    control_columns: tuple[Text, ...] = ()
    active_columns: tuple[Text, ...] = ()
    target_column: Text | None = None
    target_arm: Text | None = None
    unresolved: Text | None = None


class _RRequest(_Model):
    tables: dict[Text, Text]

    @model_validator(mode="after")
    def validate_tables(self) -> Self:
        """Require the data table and keep CSV paths inside the request."""
        if "data" not in self.tables:
            raise ValueError("R request requires a table named data")
        for value in self.tables.values():
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts or "\\" in value or path.suffix != ".csv":
                raise ValueError("R input must be a relative CSV path inside the request directory")
        return self


class REstimateRequest(_RRequest):
    """Select a declared cell/route pair or an explicit panel configuration."""

    type: Literal["estimate"]
    cell_id: Text | None = None
    route_id: Text | None = None
    configuration: RConfiguration | None = None
    parameters: EmptyObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Match tabverify's mutually exclusive estimation selectors."""
        paired = self.cell_id is not None and self.route_id is not None
        if (self.cell_id is not None or self.route_id is not None) and (not paired or self.configuration is not None):
            raise ValueError("supply a cell/route pair or a panel configuration")
        if not paired and self.configuration is None:
            raise ValueError("estimate configuration required")
        return self


class RBinaryStratum(BinaryValidationStratum):
    """Shared or arm-specific binary accuracy accepted by tabverify."""

    accuracy_by_arm: tuple[ArmAccuracy, ...] = ()
    sensitivity: Probability | None = None
    specificity: Probability | None = None

    @model_validator(mode="after")
    def require_accuracy(self) -> Self:
        """Require either shared accuracy or both randomized arms."""
        if self.accuracy_by_arm:
            if sorted(a.arm_id for a in self.accuracy_by_arm) != ["active", "control"]:
                raise ValueError("binary accuracy requires active and control arms")
        elif self.sensitivity is None or self.specificity is None:
            raise ValueError("binary stratum requires sensitivity and specificity")
        return self


class RCountStratum(CountValidationStratum):
    """Shared or arm-specific count capture accepted by tabverify."""

    capture_by_arm: tuple[ArmCapture, ...] = ()
    capture_probability: Probability | None = None

    @model_validator(mode="after")
    def require_capture(self) -> Self:
        """Require shared capture or both randomized arms."""
        if self.capture_by_arm:
            if sorted(a.arm_id for a in self.capture_by_arm) != ["active", "control"]:
                raise ValueError("count capture requires active and control arms")
        elif self.capture_probability is None:
            raise ValueError("count stratum requires capture_probability")
        return self


class RMechanismParameters(_Model):
    """Endpoint selector and exported or shared validation-stratum declarations."""

    strata: tuple[RBinaryStratum | RCountStratum, ...] = Field(min_length=1)
    endpoint_type: Literal["binary", "count"] = "binary"

    @model_validator(mode="after")
    def validate_endpoint(self) -> Self:
        """Keep the endpoint selector consistent with its typed strata."""
        expected = RBinaryStratum if self.endpoint_type == "binary" else RCountStratum
        if any(not isinstance(stratum, expected) for stratum in self.strata):
            raise ValueError("validation strata must match endpoint_type")
        return self


class RMechanismRequest(_RRequest):
    """Test declared measurement parameters, retaining strata and arm accuracy."""

    type: Literal["mechanism_parameters"]
    parameters: RMechanismParameters


class RMechanismDifferenceParameters(_Model):
    """Endpoint selector for the descriptive R arm-difference routine."""

    endpoint_type: Literal["binary", "count"] = "binary"


class RMechanismDifferenceRequest(_RRequest):
    """Descriptive legacy mechanism comparison, separate from declaration tests."""

    type: Literal["mechanism"]
    parameters: RMechanismDifferenceParameters = Field(default_factory=RMechanismDifferenceParameters)


class RMainEffect(_Model):
    """Minimal declared main effect accepted by the contact-level R routines."""

    effect_type: Literal["main"]
    input_variable: Text
    coefficient: float


class RObservationDeclaration(_Model):
    """R's contact-logistic inputs when no complete export is supplied."""

    intercept: float
    time_coefficient: float
    baseline_coefficient: float = 0
    baseline_variable: Text = "bmi_curve"
    effects: tuple[MainEffect | RMainEffect, ...] = ()
    monotone_dropout: bool = False
    observe_first_contact: bool = False
    terminal_missingness_only: bool = False


class RObservationParameters(_Model):
    """Normative observation declaration for the contact-level logistic fit."""

    declaration: Observation | RObservationDeclaration


class RICEDeclaration(RObservationDeclaration):
    """R's at-risk logistic inputs with baseline and randomized-arm effects."""

    baseline_coefficient: float
    arm_coefficients: tuple[ArmCoefficient, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def require_arms(self) -> Self:
        """Require one coefficient for each randomized arm."""
        if sorted(a.arm_id for a in self.arm_coefficients) != ["active", "control"]:
            raise ValueError("ICE coefficients require active and control arms")
        return self


class RObservationRequest(_RRequest):
    """Estimate each declared observation-process coefficient."""

    type: Literal["observation_coefficients"]
    parameters: RObservationParameters


class RICEParameters(_Model):
    """Normative intercurrent-event declaration for the at-risk logistic fit."""

    declaration: IntercurrentEvent | RICEDeclaration


class RICERequest(_RRequest):
    """Estimate each declared intercurrent-event coefficient."""

    type: Literal["ice_coefficients"]
    parameters: RICEParameters


class RLossParameters(_Model):
    """Declared annual hazard and the process/cell censoring multipliers."""

    hazard: Annotated[float, Field(gt=0)] | None
    process_multiplier: Annotated[float, Field(gt=0)] = 1
    cell_multiplier: Annotated[float, Field(gt=0)] = 1
    cluster: bool = False


class RLossRequest(_RRequest):
    """Test the pooled loss count against its declared rate and exposure."""

    type: Literal["loss_rate"]
    parameters: RLossParameters


class RCalendarParameters(_Model):
    """Declared calendar log-hazard offset and accrual window."""

    offset: float
    accrual_days: Annotated[float, Field(gt=0)]


class RCalendarRequest(_RRequest):
    """Test the calendar Cox coefficient against the declared offset."""

    type: Literal["calendar_coefficient"]
    parameters: RCalendarParameters


class RPoolTruthRequest(_RRequest):
    """Reserved TrialDev pool-truth request."""

    type: Literal["pool_truth"]
    parameters: EmptyObject = Field(default_factory=dict)


RPayload = Annotated[
    REstimateRequest
    | RMechanismRequest
    | RMechanismDifferenceRequest
    | RObservationRequest
    | RICERequest
    | RLossRequest
    | RCalendarRequest
    | RPoolTruthRequest,
    Field(discriminator="type", json_schema_extra={"$id": "trialagentbench.properties.r_payload/v1"}),
]


class REstimate(_Model):
    """An R estimate and interval, or an explicitly unsuccessful fit.

    Notes
    -----
    Fit statuses are ``ok``, ``failed`` and ``unavailable``. They are separate
    from property verdicts. Failed or unavailable fits carry null estimates.
    """

    type: Literal["estimate"]
    status: Literal["ok", "failed", "unavailable"]
    estimate: float | None
    lower: float | None
    upper: float | None
    se: Annotated[float, Field(ge=0)] | None = None
    message: str
    versions: dict[str, str]

    @model_validator(mode="after")
    def validate_fit(self) -> Self:
        """Distinguish fitted intervals from unavailable numerical results."""
        values = (self.estimate, self.lower, self.upper)
        if self.status == "ok":
            if any(value is None for value in values):
                raise ValueError("ok fit requires estimate, lower and upper")
            if self.lower is not None and self.upper is not None and self.lower > self.upper:
                raise ValueError("fit interval is reversed")
        elif any(value is not None for value in (*values, self.se)):
            raise ValueError("unsuccessful fit requires null numerical results")
        return self


class RMechanismParameter(_Model):
    """Exact binomial declaration test with pooled numerator and denominator."""

    parameter: Literal["selection", "sensitivity", "specificity", "capture"]
    arm: Text | None
    stratum: Text | None
    numerator: NonnegativeInt
    n: NonnegativeInt
    realized: Probability | None
    declared: Probability
    p: Probability | None
    availability: Literal["available", "unavailable"]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Retain unavailable zero denominators without fabricating a p-value."""
        if self.numerator > self.n:
            raise ValueError("numerator exceeds n")
        available = self.n > 0
        if (self.availability == "available") != available or any(
            (value is not None) != available for value in (self.realized, self.p)
        ):
            raise ValueError("binomial availability and numerical values must match the denominator")
        return self


class RModelCoefficient(_Model):
    """One fitted process coefficient and its declared-value Wald test."""

    parameter: Text
    declared: float
    estimate: float | None
    se: Annotated[float, Field(gt=0)] | None
    z: float | None
    p: Probability | None
    availability: Literal["available", "unavailable"]

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        """Keep unsupported coefficients explicitly unavailable."""
        available = self.availability == "available"
        if any((value is not None) != available for value in (self.estimate, self.se, self.z, self.p)):
            raise ValueError("coefficient availability must match its numerical values")
        return self


class RLossParameter(_Model):
    """Exact Poisson declaration test retaining pooled loss count and exposure."""

    parameter: Literal["loss_rate"]
    numerator: NonnegativeInt
    exposure_years: Annotated[float, Field(gt=0)]
    realized: Annotated[float, Field(ge=0)]
    declared: Annotated[float, Field(gt=0)]
    p: Probability
    availability: Literal["available"]


class RMechanismDifference(_Model):
    """Legacy descriptive arm-difference component; no declared-value test."""

    parameter: Literal["sensitivity", "specificity", "capture"]
    p: Probability


class RMechanism(_Model):
    """Shared R envelope for mechanism, logistic, loss and calendar tests.

    Notes
    -----
    Envelope availability means components exist. Individual components can
    still be unavailable; an unavailable coefficient is never a passing test.
    """

    type: Literal["mechanism"]
    p_diff: Probability | None = None
    components: tuple[RMechanismParameter | RModelCoefficient | RLossParameter | RMechanismDifference, ...]
    availability: Literal["available", "unavailable"]
    message: str

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        """Match the R component-presence meaning of envelope availability."""
        if (self.availability == "available") != bool(self.components):
            raise ValueError("mechanism availability must match component presence")
        return self


class PoolContrast(_Model):
    """One named pool-truth contrast on its declared effect scale."""

    contrast_id: Text
    estimate: float


class RPoolTruth(_Model):
    """Independent contrasts reconstructed from a released TrialDev pool."""

    type: Literal["pool_truth"]
    contrasts: tuple[PoolContrast, ...]


RResult = Annotated[
    REstimate | RMechanism | RPoolTruth,
    Field(discriminator="type", json_schema_extra={"$id": "trialagentbench.properties.r_result/v1"}),
]


class DeterministicResult(_Model):
    """Exact check counts and internal failure evidence for one cell."""

    schema_id: Literal["trialagentbench.properties.deterministic_result/v1"] = (
        "trialagentbench.properties.deterministic_result/v1"
    )
    check_id: CheckId
    cell_ref: Text
    n_checked: NonnegativeInt
    n_failed: NonnegativeInt
    failures: tuple[dict[str, JsonValue], ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Reject an impossible exact-check count."""
        if self.n_failed > self.n_checked:
            raise ValueError("n_failed exceeds n_checked")
        return self


class StatisticalPart(_Model):
    """One named falsification-test part, including unavailable numerical values."""

    name: Text
    statistic: float | None
    p: Probability | None
    n: NonnegativeInt
    parameter: Text | None = None
    arm_id: Text | None = None
    stratum: Text | None = None
    holm_p: Probability | None = None
    route_id: Text | None = None
    task_key: Text | None = None
    effect_index: Annotated[int, Field(ge=0, le=3, strict=True)] | None = None
    economics_point: EconomicsPoint | None = None
    branch: Literal["shift", "interval", "not_inverted", "benign"] | None = None
    n_ok: NonnegativeInt | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Bind task/effect dimensions and keep fitted counts within their denominator."""
        if (self.task_key is None) != (self.effect_index is None):
            raise ValueError("task_key and effect_index must be supplied together")
        if self.n_ok is not None and self.n_ok > self.n:
            raise ValueError("n_ok exceeds n")
        if self.holm_p is not None and self.p is None:
            raise ValueError("part holm_p requires a raw p-value")
        return self


class TierOnlyVerdict(_Model):
    """Public tier-only projection without parameter dimensions or numerical evidence."""

    cell: Text
    family: Family
    verdict: Status
    holm_p: Probability | None


class CheckResult(_Model):
    """One cell-level check or descriptive task-level check result.

    Notes
    -----
    Deterministic results have no statistical family or adjusted p-value.
    Null part values retain unavailable subtests without fabricating p-values.
    Statistical adjustment and publication decisions are computed by the runner.
    MECH and process checks retain parameter dimensions and per-part Holm p-values;
    each family spans all parameter tests across the split, without cell aggregation.
    REC retains seven parts per route and both Bonferroni aggregation levels.
    CONSEQ retains its four branches before the cell composite. REGRET uses one
    result per cell and economics point; Holm spans those cell/point results.
    Effect indices are zero-based. Task results are descriptive and unadjusted.
    """

    schema_id: Literal["trialagentbench.properties.check_result/v1"] = "trialagentbench.properties.check_result/v1"
    check_id: CheckId
    scope: Literal["cell", "task"]
    scope_id: Text
    family: Family | None
    parts: tuple[StatisticalPart, ...]
    route_p: dict[Text, Probability | None]
    cell_p: Probability | None
    holm_p: Probability | None
    status: Status

    @model_validator(mode="after")
    def validate_check_dimensions(self) -> Self:
        """Require each statistical check's family, dimensions and aggregation units."""
        families: dict[str, Family] = {
            "L1-MECH": "F-MECH",
            "L1-OBS": "F-OBS",
            "L1-ICE": "F-ICE",
            "L1-CENS": "F-CENS",
            "L1-CAL": "F-CAL",
            "L1-REC": "F-REC",
            "L1-CONSEQ-A3": "F-CONSEQ",
            "L1-REGRET-A2": "F-REGRET",
        }
        if self.scope == "task" or self.family is None:
            if (
                self.family is not None
                or self.route_p
                or self.cell_p is not None
                or self.holm_p is not None
                or any(part.holm_p is not None for part in self.parts)
            ):
                raise ValueError("descriptive and deterministic results cannot carry aggregate p-values or a family")
            if self.scope == "cell" and self.check_id in {"L1-MECH", "L1-REC", "L1-CONSEQ-A3", "L1-REGRET-A2"}:
                raise ValueError("statistical cell result requires its family")
            return self
        if self.family != families.get(self.check_id):
            raise ValueError("statistical family must match check_id")
        if not self.parts:
            raise ValueError("statistical cell result requires parts")
        if self.check_id in {"L1-MECH", "L1-OBS", "L1-ICE", "L1-CENS", "L1-CAL"}:
            self.validate_parameter_dimensions()
        elif any(part.holm_p is not None for part in self.parts):
            raise ValueError("per-part holm_p is only defined for mechanism and process checks")
        if self.check_id in {"L1-REC", "L1-CONSEQ-A3", "L1-REGRET-A2"}:
            if any(part.route_id is None or part.n_ok is None for part in self.parts):
                raise ValueError("recovery and consequence parts require route_id and n_ok")
        if self.check_id == "L1-REC":
            self.validate_recovery_dimensions()
        elif self.route_p:
            raise ValueError("route_p is only defined for L1-REC")
        if self.check_id == "L1-CONSEQ-A3":
            self.validate_consequence_dimensions()
        elif any(part.branch is not None for part in self.parts):
            raise ValueError("branch is only defined for L1-CONSEQ-A3")
        if self.check_id == "L1-REGRET-A2":
            if len(self.parts) != 1 or self.parts[0].economics_point is None:
                raise ValueError("L1-REGRET-A2 requires one part per cell and economics_point")
            if self.parts[0].task_key is not None:
                raise ValueError("L1-REGRET-A2 pools all tasks at its economics_point")
        elif any(part.economics_point is not None for part in self.parts):
            raise ValueError("economics_point is only defined for L1-REGRET-A2")
        return self

    def validate_parameter_dimensions(self) -> None:
        """Identify every declared parameter test before split-wide family adjustment."""
        if any(part.parameter is None for part in self.parts):
            raise ValueError(f"{self.check_id} parts require parameter")
        if self.cell_p is not None or self.holm_p is not None:
            raise ValueError("mechanism and process checks use per-part Holm adjustment, not cell aggregation")
        if self.check_id == "L1-MECH":
            for part in self.parts:
                if part.parameter not in {"selection_share", "sensitivity", "specificity", "capture"}:
                    raise ValueError("L1-MECH requires a declared mechanism parameter")
                if part.stratum is None:
                    raise ValueError("L1-MECH parts require stratum")
                if (part.arm_id is None) != (part.parameter == "selection_share"):
                    raise ValueError("L1-MECH requires arm_id for sensitivity, specificity and capture only")
        dimensions = [(part.parameter, part.arm_id, part.stratum) for part in self.parts]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError(f"{self.check_id} parameter test dimensions must be distinct")

    def validate_recovery_dimensions(self) -> None:
        """Retain each route's pooled bias, four task biases, coverage and support."""
        routes = {part.route_id for part in self.parts}
        if self.route_p.keys() != routes:
            raise ValueError("route_p must identify every L1-REC route exactly")
        for route in routes:
            parts = [part for part in self.parts if part.route_id == route]
            effects = [part for part in parts if part.name == "effect_bias"]
            pooled = [part for part in parts if part.name != "effect_bias"]
            if (
                len(parts) != 7
                or sorted(part.name for part in pooled) != ["bias", "coverage", "support"]
                or {part.effect_index for part in effects} != {0, 1, 2, 3}
                or len({part.task_key for part in effects}) != 4
                or any(part.task_key is None for part in effects)
                or any(part.task_key is not None for part in pooled)
            ):
                raise ValueError("L1-REC requires seven parts per route with four distinct task/effect biases")

    def validate_consequence_dimensions(self) -> None:
        """Retain task shifts, interval evidence, recovery/MSE inversion and the benign twin."""
        shifts = [part for part in self.parts if part.branch == "shift"]
        if (
            {part.branch for part in self.parts} != {"shift", "interval", "not_inverted", "benign"}
            or len(shifts) != 4
            or {part.effect_index for part in shifts} != {0, 1, 2, 3}
            or len({part.task_key for part in shifts}) != 4
            or any(part.task_key is None for part in shifts)
            or sorted(part.name for part in self.parts if part.branch == "not_inverted") != ["mse", "recovery"]
        ):
            raise ValueError("L1-CONSEQ-A3 requires four task shifts, interval, not_inverted recovery/mse and benign")


class PublicSplitEvidence(_Model):
    """Counts, margin and digests supporting a narrowly worded public claim."""

    n: NonnegativeInt
    n_verified: NonnegativeInt
    margin: float | str | None
    digests: tuple[Sha256, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require verified counts to fit within the claim's denominator."""
        if self.n_verified > self.n:
            raise ValueError("n_verified exceeds n")
        return self


class WithheldSplitEvidence(_Model):
    """Withheld claim evidence containing only aggregate counts and digests."""

    n: NonnegativeInt
    n_verified: NonnegativeInt
    digests: tuple[Sha256, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require verified counts to fit within the withheld denominator."""
        if self.n_verified > self.n:
            raise ValueError("n_verified exceeds n")
        return self


class Verdict(_Model):
    """Positive public claim row with its supporting counts and margins.

    Notes
    -----
    The retained class/aggregate statistics support machine consumers. Only
    passing claims enter this public contract; unavailable and failed checks
    use ``CheckResult`` in the internal report. No cell or task identifier is
    accepted, including within withheld evidence.
    """

    schema_id: Literal["trialagentbench.properties.verdict/v1"] = "trialagentbench.properties.verdict/v1"
    criterion_id: Text
    scope: Literal["class", "public_aggregate"]
    claim: Text
    how_verified: Text
    public_split: PublicSplitEvidence | None
    withheld_split: WithheldSplitEvidence | None
    status: Literal["pass"] = "pass"
    statistic: float | None
    interval: tuple[float, float] | None
    n: NonnegativeInt


class PublicRunFiles(_Model):
    """Exact public export filenames and their content digests."""

    class_summary_csv: FileDigest
    verdicts_json: FileDigest
    figure_svg: FileDigest
    figure_pdf: FileDigest
    run_json: FileDigest
    sha256sums: FileDigest

    @model_validator(mode="after")
    def validate_filenames(self) -> Self:
        """Require the public export's six specified relative filenames."""
        expected = {
            "class_summary_csv": "class_summary.csv",
            "verdicts_json": "verdicts.json",
            "figure_svg": "figure.svg",
            "figure_pdf": "figure.pdf",
            "run_json": "run.json",
            "sha256sums": "SHA256SUMS",
        }
        if any(getattr(self, field).path != path for field, path in expected.items()):
            raise ValueError("public export file paths differ from the declared filenames")
        return self


class PublicRunExport(_Model):
    """Completed public export inventory, with class-level positive claims only.

    Notes
    -----
    This handoff envelope describes the completed directory. It is separate
    from the directory's ``run.json`` member, so no self-digest is required.
    """

    schema_id: Literal["trialagentbench.properties.public_run_export/v1"] = (
        "trialagentbench.properties.public_run_export/v1"
    )
    release_id: Text
    split: Literal["public"] = "public"
    files: PublicRunFiles
    verdicts: tuple[Verdict, ...]


class InternalAttestation(_Model):
    """Private run attestation with the same claim schema and input-manifest digest.

    Notes
    -----
    Per-cell and per-task details remain outside this export, in
    ``internal/detail/``. The input manifest binds private evidence; its contents
    and private paths are not duplicated in claim rows.
    """

    schema_id: Literal["trialagentbench.properties.internal_attestation/v1"] = (
        "trialagentbench.properties.internal_attestation/v1"
    )
    release_id: Text
    split: Split
    run_input_manifest_sha256: Sha256
    verdicts: tuple[Verdict, ...]
    files: PublicRunFiles
