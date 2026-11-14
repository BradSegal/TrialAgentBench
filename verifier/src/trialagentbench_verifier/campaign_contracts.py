"""Compatibility import for ``trialagentbench_verifier.evaluation.campaign_contracts``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import campaign_contracts as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.campaign_contracts import (
        BudgetCensoringV1 as BudgetCensoringV1,
        CampaignAnalysisV1 as CampaignAnalysisV1,
        CampaignConfigurationV1 as CampaignConfigurationV1,
        CampaignDispatchV1 as CampaignDispatchV1,
        CampaignModelV1 as CampaignModelV1,
        CampaignOperationsV1 as CampaignOperationsV1,
        CampaignPanelV1 as CampaignPanelV1,
        CampaignPlanV1 as CampaignPlanV1,
        CampaignRequestPricingV1 as CampaignRequestPricingV1,
        CampaignRunV1 as CampaignRunV1,
        CampaignTaskPriorityV1 as CampaignTaskPriorityV1,
        CensoredRunV1 as CensoredRunV1,
        Digest as Digest,
        Identifier as Identifier,
        LeaderboardRowV1 as LeaderboardRowV1,
        MATERIAL_CENSORING_DIFFERENCE as MATERIAL_CENSORING_DIFFERENCE,
        ModelCensoringV1 as ModelCensoringV1,
        WEB_ENABLED_ROUTE as WEB_ENABLED_ROUTE,
        _Model as _Model,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
