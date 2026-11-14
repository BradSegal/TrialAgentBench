"""Compatibility import for ``trialagentbench_verifier.evaluation.trialeval_inputs``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import trialeval_inputs as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.trialeval_inputs import (
        JsonArray as JsonArray,
        JsonObject as JsonObject,
        JsonScalar as JsonScalar,
        JsonValue as JsonValue,
        TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS as TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS,
        TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS as TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS,
        _ANALYSIS_DATA_FILENAMES as _ANALYSIS_DATA_FILENAMES,
        canonical_trialeval_workspace_path as canonical_trialeval_workspace_path,
        canonical_trialeval_workspace_paths as canonical_trialeval_workspace_paths,
        sanitize_trialeval_agent_analysis_plan_payload as sanitize_trialeval_agent_analysis_plan_payload,
        sanitize_trialeval_agent_endpoint_definition_payload as sanitize_trialeval_agent_endpoint_definition_payload,
        sanitize_trialeval_agent_json_payload as sanitize_trialeval_agent_json_payload,
        sanitize_trialeval_agent_json_text as sanitize_trialeval_agent_json_text,
        sanitize_trialeval_agent_task_payload as sanitize_trialeval_agent_task_payload,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
