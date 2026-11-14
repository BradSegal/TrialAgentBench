"""Compatibility import for ``trialagentbench_verifier.runtime.run_contracts``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.runtime import run_contracts as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.runtime.run_contracts import (
        DecodingConfigV1 as DecodingConfigV1,
        ExperimentConditionV1 as ExperimentConditionV1,
        MemoryStrategyV1 as MemoryStrategyV1,
        ProcedureAssistanceV1 as ProcedureAssistanceV1,
        ProviderModelCapabilityV1 as ProviderModelCapabilityV1,
        ProviderRetryPolicyV1 as ProviderRetryPolicyV1,
        ProviderUnavailabilityReceiptV1 as ProviderUnavailabilityReceiptV1,
        ReasoningConfigV1 as ReasoningConfigV1,
        ReasoningEffortV1 as ReasoningEffortV1,
        RoutingConfigV1 as RoutingConfigV1,
        ToolChoiceV1 as ToolChoiceV1,
        TrialEvalAgentStatusV1 as TrialEvalAgentStatusV1,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
