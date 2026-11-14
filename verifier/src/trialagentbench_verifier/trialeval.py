"""Compatibility import for ``trialagentbench_verifier.evaluation.trialeval``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import trialeval as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.trialeval import (
        AttemptStatus as AttemptStatus,
        ComponentId as ComponentId,
        ComponentStatus as ComponentStatus,
        Conclusion as Conclusion,
        Context as Context,
        DecisionKind as DecisionKind,
        DeclaredEffectMeasure as DeclaredEffectMeasure,
        ResultEffectMeasure as ResultEffectMeasure,
        TrialEvalDataIntegrityKeyV1 as TrialEvalDataIntegrityKeyV1,
        TrialEvalDecisionRuleV1 as TrialEvalDecisionRuleV1,
        TrialEvalGradingKeyRecordV1 as TrialEvalGradingKeyRecordV1,
        TrialEvalGradingPlanV1 as TrialEvalGradingPlanV1,
        TrialEvalOutcomeVerificationError as TrialEvalOutcomeVerificationError,
        TrialEvalScoreInputV1 as TrialEvalScoreInputV1,
        TrialEvalTargetScoreInputV1 as TrialEvalTargetScoreInputV1,
        TrialEvalTargetV1 as TrialEvalTargetV1,
        VerifiedTrialEvalComponentV1 as VerifiedTrialEvalComponentV1,
        VerifiedTrialEvalOutcomeV1 as VerifiedTrialEvalOutcomeV1,
        VerifiedTrialEvalPlanExecutionV1 as VerifiedTrialEvalPlanExecutionV1,
        VerifiedTrialEvalTargetOutcomeV1 as VerifiedTrialEvalTargetOutcomeV1,
        _ConsequenceEconomics as _ConsequenceEconomics,
        _ConsequenceReference as _ConsequenceReference,
        _Interval as _Interval,
        _Model as _Model,
        _ORACLE_EFFECT_MEASURES as _ORACLE_EFFECT_MEASURES,
        _PLAN_CONTEXTS as _PLAN_CONTEXTS,
        _PointResult as _PointResult,
        _PointTarget as _PointTarget,
        _Question as _Question,
        _RATIO_EFFECT_MEASURES as _RATIO_EFFECT_MEASURES,
        _Reference as _Reference,
        _Result as _Result,
        _comparison as _comparison,
        _component as _component,
        _estimand_core_matches as _estimand_core_matches,
        _expected_conclusion as _expected_conclusion,
        _finite_decimal as _finite_decimal,
        _interval_loss as _interval_loss,
        _load_record as _load_record,
        _metrics as _metrics,
        _missing_origin as _missing_origin,
        _question_matches as _question_matches,
        _result_scale_matches as _result_scale_matches,
        _signal_loss as _signal_loss,
        _verify_decision as _verify_decision,
        _verify_target as _verify_target,
        trialeval_decision_targets_v1 as trialeval_decision_targets_v1,
        trialeval_decisions_v1 as trialeval_decisions_v1,
        trialeval_plan_execution_v1 as trialeval_plan_execution_v1,
        verify_trialeval_outcome_v1 as verify_trialeval_outcome_v1,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
