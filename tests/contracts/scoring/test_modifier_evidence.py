"""Tests for the public method-modifier vocabulary."""

from trialagentbench.contracts.scoring.modifier_evidence import METHOD_MODIFIERS_V1


def test_flexible_model_form_is_a_public_method_modifier() -> None:
    assert "flexible_model_form" in METHOD_MODIFIERS_V1


def test_ph_robust_fixed_horizon_is_a_public_method_modifier() -> None:
    assert "ph_robust_fixed_horizon" in METHOD_MODIFIERS_V1
