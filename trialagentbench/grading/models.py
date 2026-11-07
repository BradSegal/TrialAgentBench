"""Scientific reference and target contracts shared with benchmark construction."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AnalysisMethodResultKindV1 = Literal[
    "numeric_point",
    "numeric_interval",
    "numeric_vector",
    "statistical_test",
    "sensitivity_set",
    "identification_bound",
    "limitation",
    "abstention",
    "decision",
]
_NON_NUMERIC_RESULT_KINDS_V1 = frozenset({"limitation", "abstention", "decision", "statistical_test"})


def result_kind_requires_procedure_v1(result_kind: str) -> bool:
    """Return whether a result kind requires a numerical estimator procedure."""

    return result_kind not in _NON_NUMERIC_RESULT_KINDS_V1


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NumericalAcceptanceEnvelopeV1(_ContractModel):
    """Prospective numerical equivalence bound for an accepted route."""

    schema_id: Literal["ariadne.numerical_acceptance_envelope/v1"] = "ariadne.numerical_acceptance_envelope/v1"
    reporting_decimal_places: int = Field(ge=0, le=15)
    independent_point_max_abs_difference: float = Field(ge=0)
    independent_interval_endpoint_max_abs_difference: float = Field(ge=0)
    public_verification_id: str = Field(min_length=1)
    independent_verification_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _independent_evidence(self) -> NumericalAcceptanceEnvelopeV1:
        if len(set(self.independent_verification_ids)) != len(self.independent_verification_ids):
            raise ValueError("independent_verification_ids must be unique")
        return self

    @property
    def point_absolute_tolerance(self) -> float:
        """Return the point-estimate rounding and implementation tolerance."""

        return 0.5 * 10.0 ** (-self.reporting_decimal_places) + self.independent_point_max_abs_difference

    @property
    def interval_endpoint_absolute_tolerance(self) -> float:
        """Return the interval-endpoint rounding and implementation tolerance."""

        return 0.5 * 10.0 ** (-self.reporting_decimal_places) + self.independent_interval_endpoint_max_abs_difference


class RouteSignatureV1(_ContractModel):
    """Exact question and complete method identity declared for grading."""

    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = Field(default=None, gt=0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    analysis_unit: Literal["participant", "cluster"] = "participant"
    population_basis: Literal["finite_population", "superpopulation"] = "finite_population"
    identification_status: Literal[
        "point_identified",
        "identified_under_assumptions",
        "partially_identified",
    ] = "point_identified"
    identification_assumption_ids: tuple[str, ...] = ()
    null_value: float = 0.0
    analysis_method_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_semantics(self) -> RouteSignatureV1:
        for name, values in (
            ("intercurrent_event_strategy_ids", self.intercurrent_event_strategy_ids),
            ("identification_assumption_ids", self.identification_assumption_ids),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        return self


class EstimatorProcedureV1(_ContractModel):
    """Complete executable identity of one numeric analysis procedure."""

    analysis_unit: Literal["participant", "cluster"]
    grouping: Literal["randomized_arm", "received_treatment", "validation_stratum"]
    partition: Literal[
        "none",
        "analysis_period",
        "reference_population",
        "validation_stratum",
        "monitoring_look",
    ]
    refitting: Literal[
        "none",
        "full_pipeline_bootstrap",
        "delete_group_jackknife",
        "joint_likelihood",
        "repeated_monitoring",
    ]
    variance: str = Field(min_length=1)
    degrees_of_freedom: Literal[
        "normal_reference",
        "independent_units_minus_one",
        "empirical_resampling",
        "sequential_reference",
        "not_applicable",
    ]
    transformation: str = Field(min_length=1)
    combination: Literal[
        "treated_minus_control",
        "treated_ratio_control",
        "identified_set",
        "statistical_test",
        "categorical_conclusion",
    ]
    confidence_level: float | None = Field(default=None, gt=0.0, lt=1.0)
    arm_quantities: tuple[str, ...]
    contrast_quantity: str = Field(min_length=1)
    fitting_specification: str = Field(
        default="Use the complete fitting procedure declared by the owning analysis method.",
        min_length=1,
    )
    resampling_unit: Literal["participant", "cluster"] | None = None
    resampling_count: int | Literal["all_independent_units"] | None = None
    resampling_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    resampling_rule: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _coherent_uncertainty(self) -> EstimatorProcedureV1:
        if len(set(self.arm_quantities)) != len(self.arm_quantities):
            raise ValueError("estimator arm quantities must be unique")
        interval_estimate = self.combination in {"treated_minus_control", "treated_ratio_control"}
        if interval_estimate != (self.confidence_level is not None):
            raise ValueError("contrast estimators require exactly one confidence level")
        if self.partition == "analysis_period" and self.analysis_unit != "cluster":
            raise ValueError("period-partitioned procedures require cluster analysis units")
        resampling_values = (
            self.resampling_unit,
            self.resampling_count,
            self.resampling_rule,
        )
        if any(value is not None for value in resampling_values) != all(
            value is not None for value in resampling_values
        ):
            raise ValueError("resampling unit, count, and rule must be supplied together")
        if self.resampling_seed is not None and self.resampling_count is None:
            raise ValueError("resampling seeds require a complete resampling contract")
        if isinstance(self.resampling_count, int) and self.resampling_count < 2:
            raise ValueError("numeric resampling counts must be at least two")
        return self


class NumericPointTargetV1(_ContractModel):
    kind: Literal["numeric_point"] = "numeric_point"
    value: float
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1
    require_confidence_interval: bool
    confidence_interval_lower: float | None = None
    confidence_interval_upper: float | None = None

    @model_validator(mode="after")
    def _complete_interval(self) -> NumericPointTargetV1:
        endpoints = (self.confidence_interval_lower, self.confidence_interval_upper)
        if (endpoints[0] is None) != (endpoints[1] is None):
            raise ValueError("target confidence interval endpoints must be supplied together")
        if self.require_confidence_interval != (endpoints[0] is not None):
            raise ValueError(
                "require_confidence_interval must equal the presence of target confidence interval endpoints"
            )
        if endpoints[0] is not None and endpoints[1] is not None and endpoints[0] > endpoints[1]:
            raise ValueError("target confidence interval lower endpoint cannot exceed upper endpoint")
        return self


class NumericIntervalTargetV1(_ContractModel):
    kind: Literal["numeric_interval"] = "numeric_interval"
    lower: float
    upper: float
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1

    @model_validator(mode="after")
    def _ordered(self) -> NumericIntervalTargetV1:
        if self.lower > self.upper:
            raise ValueError("target interval lower endpoint cannot exceed upper endpoint")
        return self


class NamedNumericValueV1(_ContractModel):
    name: str = Field(min_length=1)
    value: float


class NumericVectorTargetV1(_ContractModel):
    kind: Literal["numeric_vector"] = "numeric_vector"
    components: tuple[NamedNumericValueV1, ...] = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    acceptance_envelope: NumericalAcceptanceEnvelopeV1

    @model_validator(mode="after")
    def _unique_components(self) -> NumericVectorTargetV1:
        names = tuple(component.name for component in self.components)
        if len(set(names)) != len(names):
            raise ValueError("vector component names must be unique")
        return self


class StatisticalTestTargetV1(_ContractModel):
    kind: Literal["statistical_test"] = "statistical_test"
    p_value: float = Field(ge=0, le=1)
    reject_null: bool
    acceptance_envelope: NumericalAcceptanceEnvelopeV1


class CategoricalTargetV1(_ContractModel):
    kind: Literal["categorical"] = "categorical"
    credit_eligible_codes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> CategoricalTargetV1:
        if len(set(self.credit_eligible_codes)) != len(self.credit_eligible_codes):
            raise ValueError("credit_eligible_codes must be unique")
        return self


ScoringTargetV1 = Annotated[
    NumericPointTargetV1
    | NumericIntervalTargetV1
    | NumericVectorTargetV1
    | StatisticalTestTargetV1
    | CategoricalTargetV1,
    Field(discriminator="kind"),
]


class AnalysisMethodBindingV1(_ContractModel):
    """Intrinsic properties selected by one canonical analysis method ID."""

    analysis_method_id: str = Field(min_length=1)
    method_family_id: str | None = Field(default=None, min_length=1)
    estimator_family: str = Field(min_length=1)
    result_kind: AnalysisMethodResultKindV1
    uncertainty_method: str = Field(min_length=1)
    sensitivity_parameters: tuple[float, ...] = ()
    design_modifiers: tuple[str, ...] = ()
    procedure: EstimatorProcedureV1 | None = None

    @model_validator(mode="after")
    def _canonical_method(self) -> AnalysisMethodBindingV1:
        if tuple(sorted(set(self.design_modifiers))) != self.design_modifiers:
            raise ValueError("design_modifiers must be sorted and unique")
        if tuple(sorted(set(self.sensitivity_parameters))) != self.sensitivity_parameters:
            raise ValueError("sensitivity_parameters must be sorted and unique")
        if bool(self.sensitivity_parameters) != (self.result_kind == "sensitivity_set"):
            raise ValueError("sensitivity parameters are required exactly for sensitivity-set methods")
        numeric = result_kind_requires_procedure_v1(self.result_kind)
        if numeric != (self.procedure is not None):
            raise ValueError("numeric methods require exactly one complete estimator procedure")
        return self


class CreditEligibleScoringRouteV1(_ContractModel):
    route_id: str = Field(min_length=1)
    signature: RouteSignatureV1
    method: AnalysisMethodBindingV1
    favorable_direction: Literal["higher", "lower", "neither"]
    required_identification_assumptions: tuple[str, ...] = Field(min_length=1)
    required_diagnostics: tuple[str, ...] = ()
    planning_calculator_id: str | None = Field(default=None, min_length=1)
    target: ScoringTargetV1

    @model_validator(mode="after")
    def _canonical_diagnostics(self) -> CreditEligibleScoringRouteV1:
        if self.signature.analysis_method_id != self.method.analysis_method_id:
            raise ValueError("route signature and intrinsic method record must use one analysis method ID")
        for name, values in (
            ("required_identification_assumptions", self.required_identification_assumptions),
            ("required_diagnostics", self.required_diagnostics),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        if not _method_accepts_target(self.method.result_kind, self.target.kind):
            raise ValueError("intrinsic method result kind is incompatible with the scoring target")
        return self


class ExcludedScoringRouteV1(_ContractModel):
    """One prospectively assessed route that is not eligible for credit."""

    route_id: str = Field(min_length=1)
    signature: RouteSignatureV1
    exclusion_code: Literal[
        "wrong_clinical_question",
        "unsupported_identification",
        "missing_public_evidence",
        "failed_operating_characteristics",
        "unavailable_implementation",
        "duplicate_participant_semantics",
    ]


def _method_accepts_target(method_kind: AnalysisMethodResultKindV1, target_kind: str) -> bool:
    """Return whether a complete method record can produce the stored target shape."""

    compatible_targets = {
        "numeric_point": {"numeric_point"},
        "numeric_interval": {"numeric_interval"},
        "numeric_vector": {"numeric_vector"},
        "statistical_test": {"statistical_test"},
        "sensitivity_set": {"numeric_interval", "numeric_vector"},
        "identification_bound": {"numeric_interval"},
        "limitation": {"categorical"},
        "abstention": {"categorical"},
        "decision": {"categorical"},
    }
    return target_kind in compatible_targets[method_kind]


class ValidatedScoringKeyV1(_ContractModel):
    """Portable evaluator key produced by benchmark construction."""

    schema_id: Literal["ariadne.scoring_key/v1"] = "ariadne.scoring_key/v1"
    release_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    credit_eligible_routes: tuple[CreditEligibleScoringRouteV1, ...] = Field(min_length=1)
    excluded_routes: tuple[ExcludedScoringRouteV1, ...] = ()
    prespecified_analysis_method_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _unambiguous_routes(self) -> ValidatedScoringKeyV1:
        route_ids = tuple(route.route_id for route in self.credit_eligible_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("credit-eligible route IDs must be unique")
        signatures = tuple(route.signature for route in self.credit_eligible_routes)
        if len(set(signatures)) != len(signatures):
            raise ValueError("credit-eligible routes must have unique primary-binding signatures")
        excluded_ids = tuple(route.route_id for route in self.excluded_routes)
        excluded_signatures = tuple(route.signature for route in self.excluded_routes)
        if len(set(excluded_ids)) != len(excluded_ids):
            raise ValueError("excluded routes must have unique route IDs")
        if len(set(excluded_signatures)) != len(excluded_signatures):
            raise ValueError("excluded routes must have unique primary-binding signatures")
        if set(route_ids) & set(excluded_ids) or set(signatures) & set(excluded_signatures):
            raise ValueError("credit-eligible and excluded routes must be disjoint")
        question_bindings = {
            (
                signature.analysis_population_id,
                signature.estimand_id,
                signature.intercurrent_event_strategy_ids,
                signature.assessment_horizon_days,
                signature.treatment_id,
                signature.comparator_id,
                signature.endpoint_id,
                signature.effect_scale,
            )
            for signature in signatures
        }
        if len(question_bindings) != 1:
            raise ValueError("credit-eligible routes must answer one precise estimand on one effect scale")
        locked_sap = self.context_tier in {"C1", "C3"}
        if locked_sap != (self.prespecified_analysis_method_id is not None):
            raise ValueError("exactly locked-SAP contexts require a prespecified analysis method")
        return self


__all__ = [
    "AnalysisMethodBindingV1",
    "CreditEligibleScoringRouteV1",
    "CategoricalTargetV1",
    "ExcludedScoringRouteV1",
    "NamedNumericValueV1",
    "NumericalAcceptanceEnvelopeV1",
    "NumericIntervalTargetV1",
    "NumericPointTargetV1",
    "NumericVectorTargetV1",
    "RouteSignatureV1",
    "result_kind_requires_procedure_v1",
    "StatisticalTestTargetV1",
    "ValidatedScoringKeyV1",
]
