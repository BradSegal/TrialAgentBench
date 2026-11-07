"""Scientific unit-label equivalence without numerical scale conversion."""

import re

_GENERIC_SCORE_POINT_UNITS = frozenset({"point", "points", "score point", "score points"})
_SCALE_CHANGING_UNIT_TOKENS = frozenset({"log", "logit", "odds", "percent", "percentage", "ratio"})
_RISK_DIFFERENCE_UNITS = frozenset(
    {
        "difference in proportions",
        "probability",
        "probability units",
        "proportion",
        "proportion difference",
        "proportion units",
        "risk difference",
        "risk difference unit",
        "risk difference units",
    }
)


def _normalized_unit_label(unit: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", unit.casefold()).split())


def _is_risk_difference_label(unit: str) -> bool:
    if _SCALE_CHANGING_UNIT_TOKENS.intersection(unit.split()):
        return False
    phrases = (
        "difference in proportion",
        "probability difference",
        "proportion difference",
        "risk difference",
    )
    return unit in _RISK_DIFFERENCE_UNITS or any(phrase in unit for phrase in phrases)


def compatible_units(reported: str, reference: str) -> bool:
    """Compare equivalent labels without converting between numerical scales."""

    reported_label = _normalized_unit_label(reported)
    reference_label = _normalized_unit_label(reference)
    if reported_label == reference_label:
        return True
    if _SCALE_CHANGING_UNIT_TOKENS.intersection(reported_label.split()):
        return False
    if _is_risk_difference_label(reported_label) and _is_risk_difference_label(reference_label):
        return True
    if not reference_label.endswith(" points"):
        return False
    if reported_label in _GENERIC_SCORE_POINT_UNITS:
        return True
    target_qualifiers = set(reference_label.split()) - {"point", "points", "score", "total"}
    reported_tokens = set(reported_label.split()) - {"point", "points", "score", "total"}
    return bool(target_qualifiers) and target_qualifiers.issubset(reported_tokens)
