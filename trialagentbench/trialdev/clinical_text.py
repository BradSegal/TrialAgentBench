"""Clinical wording at TrialDev presentation boundaries."""


def trialdev_clinical_text(text: str) -> str:
    """Qualify the study-specific arthritis composite without changing data identifiers or units."""
    return text.replace(
        "Disease Activity Score in 28 joints", "study-specific disease-activity composite modelled on DAS28-CRP"
    )
