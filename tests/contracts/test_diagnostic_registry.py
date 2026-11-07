from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_trialeval_diagnostic_dictionary,
    write_minimal_trialeval_diagnostic_dictionary,
)

from trialagentbench.contracts import trialeval_diagnostics
from trialagentbench.contracts.scoring.diagnostic_registry import (
    DiagnosticKeySpecV1,
    diagnostic_key_by_assumption_id_v1,
    diagnostic_normal_critical_value_v1,
    diagnostic_registry_path_v1,
    load_diagnostic_registry_v1,
)
from trialagentbench.contracts.trialeval_diagnostics import (
    PARTICIPANT_PROSE_FIELDS,
    participant_diagnostic_dictionary_v1,
)
from trialagentbench.trialeval.data import load_participant_diagnostic_dictionary


def test_diagnostic_registry_contains_semantics_without_family_answers() -> None:
    registry = load_diagnostic_registry_v1()

    assert registry.schema_id == "ariadne.diagnostic_registry/v1"
    assert set(registry.assumption_tier_operating_definitions) == {"A1", "A2", "A3"}
    a2 = registry.assumption_tier_operating_definitions["A2"]
    assert "default-defensible" in a2.label and "consequential decision" in a2.operative_definition
    assert "materiality" not in a2.operative_definition
    assert "does not select an estimator" in registry.assumption_tier_operating_definitions["A3"].scoring_constraint
    assert registry.diagnostic_keys["proportional_hazards"].public_key == "proportional_hazards_public"
    assert "families" not in type(registry).model_fields
    assert "status_by_family" not in type(registry).model_fields
    assert diagnostic_key_by_assumption_id_v1()["model_form"] == "model_form_public"
    ph = registry.diagnostic_keys["proportional_hazards"]
    assert ph.severity_thresholds is not None
    assert ph.severity_thresholds.metric_name == "simultaneous_lower_abs_time_varying_log_hazard_range"
    assert set(ph.severity_thresholds.decision_metric_names.values()) == {
        "simultaneous_lower_abs_time_varying_log_hazard_range"
    }
    assert ph.primary_method_change_threshold == pytest.approx(math.log(1.25))
    endpoint = registry.diagnostic_keys["endpoint_ascertainment"]
    assert endpoint.severity_thresholds is not None
    assert endpoint.severity_thresholds.decision_metric_names == {
        "stressed": "validation_discordance_fraction",
        "fragile": "validation_discordance_fraction",
        "broken": "unsupported_validation_stratum_fraction",
    }
    assert registry.diagnostic_keys["cluster_structure"].severity_thresholds is None
    assert diagnostic_normal_critical_value_v1(assumption_id="proportional_hazards", comparisons=1) == pytest.approx(
        1.9599639845
    )


def test_participant_diagnostic_dictionary_is_an_exact_item_independent_projection() -> None:
    registry = load_diagnostic_registry_v1()
    dictionary = minimal_trialeval_diagnostic_dictionary()

    assert set(dictionary.diagnostics) == {definition.public_key for definition in registry.diagnostic_keys.values()}
    ph = dictionary.diagnostics["proportional_hazards_public"]
    assert ph.required_metric_units == {"simultaneous_lower_abs_time_varying_log_hazard_range": "log_hazard_ratio"}
    dispersion = dictionary.diagnostics["dispersion_public"]
    assert dispersion.metric_units == {
        "pearson_chi_square": "chi_square",
        "pearson_variance_to_mean_ratio": "ratio",
        "residual_degrees_of_freedom": "count",
    }
    assert "log(1.25)" in ph.decision_boundary
    assert ph.participant_operation == "Fixture operation for proportional_hazards_public."
    payload = json.dumps(dictionary.model_dump(mode="json"), sort_keys=True)
    assert '"task_id"' not in payload
    assert '"result"' not in payload
    assert '"severity_thresholds"' not in payload
    assert "generating schedule" not in payload
    assert "assumption tier" not in payload
    assert "stressed" not in payload
    assert "fragile" not in payload
    assert "broken" not in payload
    assert "0.090602" not in payload
    assert diagnostic_normal_critical_value_v1(assumption_id="model_form", comparisons=4) > 1.9599639845


def test_harness_registry_holds_evaluator_semantics_but_no_participant_operations() -> None:
    """Construction owns participant operation text; the registry cannot render it into releases."""

    assert PARTICIPANT_PROSE_FIELDS == {"participant_operation"}
    assert "participant_operation" not in DiagnosticKeySpecV1.model_fields
    assert "participant_operation" not in diagnostic_registry_path_v1().read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="cover exactly the registered diagnostics"):
        participant_diagnostic_dictionary_v1({"proportional_hazards_public": "Fit the Cox model."})


def test_participant_wording_change_leaves_a_shipped_dictionary_loadable(tmp_path: Path) -> None:
    """A release shipped with other participant wording loads unchanged under the installed harness."""

    released = tmp_path / "released"
    released.mkdir()
    write_minimal_trialeval_diagnostic_dictionary(released, operation="Wording frozen at release build")
    before = (released / "diagnostic_dictionary.json").read_bytes()

    _, dictionary = load_participant_diagnostic_dictionary(released)

    assert dictionary != minimal_trialeval_diagnostic_dictionary(operation="Reworded after release")
    assert dictionary.diagnostics["dispersion_public"].participant_operation == (
        "Wording frozen at release build for dispersion_public."
    )
    assert (released / "diagnostic_dictionary.json").read_bytes() == before


def _install_registry_change(monkeypatch: pytest.MonkeyPatch, assumption_id: str, **update: object) -> None:
    registry = load_diagnostic_registry_v1()
    keys = dict(registry.diagnostic_keys)
    keys[assumption_id] = keys[assumption_id].model_copy(update=update)
    changed = registry.model_copy(update={"diagnostic_keys": keys})
    monkeypatch.setattr(trialeval_diagnostics, "load_diagnostic_registry_v1", lambda: changed)


@pytest.mark.parametrize(
    ("assumption_id", "update"),
    [
        ("proportional_hazards", {"primary_method_change_threshold": math.log(1.5)}),
        ("dispersion", {"metric_units": {"pearson_variance_to_mean_ratio": "ratio"}}),
        ("cluster_structure", {"primary_credit_policy": "always"}),
        ("model_form", {"diagnostic_scale": "hazard-ratio"}),
    ],
)
def test_evaluator_semantic_change_refuses_a_shipped_dictionary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assumption_id: str, update: dict[str, object]
) -> None:
    """Thresholds, units, credit policy and scales still bind the release to the installed evaluator."""

    write_minimal_trialeval_diagnostic_dictionary(tmp_path)
    load_participant_diagnostic_dictionary(tmp_path)
    _install_registry_change(monkeypatch, assumption_id, **update)

    with pytest.raises(ValueError, match="differs from the installed canonical registry"):
        load_participant_diagnostic_dictionary(tmp_path)
