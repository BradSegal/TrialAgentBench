"""The annotated example names every campaign setting, with values the configuration accepts."""

import json
import re
from pathlib import Path

from pydantic import BaseModel
from trialagentbench_verifier.campaign_contracts import CampaignConfigurationV1

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "campaign.annotated.jsonc"


def _unset_fields(model: BaseModel, path: str = "") -> list[str]:
    """Return every field of a configuration that the example left to its default."""
    missing = []
    for name in type(model).model_fields:
        value = getattr(model, name)
        if name not in model.model_fields_set and name not in {"schema_id", "rejection_policy"}:
            missing.append(f"{path}{name}")
        for child in value if isinstance(value, tuple) else (value,):
            if isinstance(child, BaseModel):
                missing.extend(_unset_fields(child, f"{path}{name}."))
    return missing


def test_the_annotated_example_names_every_setting_and_validates() -> None:
    text = re.sub(r"(?m)(^|\s)//.*$", "", _EXAMPLE.read_text(encoding="utf-8"))
    configuration = CampaignConfigurationV1.model_validate(json.loads(text))
    assert _unset_fields(configuration) == []
