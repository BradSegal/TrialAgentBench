"""Expected participant action rejections, separate from evaluator integrity failures."""


class TrialDevActionRejectedError(ValueError):
    """A participant request violates the available programme actions or design limits."""
