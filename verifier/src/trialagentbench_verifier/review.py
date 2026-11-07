"""Small offline review exports of independently computed verification results."""

from __future__ import annotations

import csv
import html
import json
import textwrap
from pathlib import Path
from typing import Literal

from matplotlib import rc_context
from matplotlib.figure import Figure

from trialagentbench_verifier.campaign import TrialEvalCampaignAnalysisV1
from trialagentbench_verifier.contracts import (
    PracticeVerificationReportV1,
    VerificationFailureReportV1,
    VerificationReportV1,
)
from trialagentbench_verifier.io import canonical_sha256
from trialagentbench_verifier.response import TrialDevRunRankingV1


def write_review(
    report: (
        PracticeVerificationReportV1 | VerificationReportV1 | TrialEvalCampaignAnalysisV1 | VerificationFailureReportV1
    ),
    directory: Path,
    *,
    evidence_kind: Literal["local_control", "historical", "live", "unclassified"] | None = None,
) -> None:
    """Export a linked HTML report, accessible CSV tables and applicable SVG.

    Parameters
    ----------
    report
        Result freshly produced by the verifier's existing public operation.
    directory
        Output directory under the same custody as the verified inputs.
    evidence_kind
        Explicit descriptive classification for a campaign, checked against
        its provenance by the reviewer; this label does not authenticate it.

    Notes
    -----
    Rendering adds no verification claim. No model code, simulation or network
    operation runs here. Campaign uncertainty retains independent-world counts;
    disclosed fixtures are never presented as model rankings.
    """
    if isinstance(report, TrialEvalCampaignAnalysisV1) and evidence_kind is None:
        raise ValueError("campaign review requires an explicit evidence kind")
    directory.mkdir(parents=True, exist_ok=False)
    payload = report.model_dump(mode="json")
    digest = canonical_sha256(payload)
    scorer_json = (
        report.scorer.model_dump_json()
        if isinstance(report, (VerificationReportV1, TrialEvalCampaignAnalysisV1))
        else ""
    )
    (directory / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    plot_rows: list[tuple[str, float, float | None, float | None, int | None, int | None]] = []
    headings: tuple[str, ...]
    rows: list[tuple[str, ...]]
    if isinstance(report, TrialEvalCampaignAnalysisV1):
        title = f"Scheduled campaign mean loss — {evidence_kind}"
        scope = (
            f"Analysis status: {report.scientific_analysis_status}. "
            "These are results from the bound campaign archive, not a declaration of current official standing. "
            "Completion counts retain all scheduled primaries. Observed-case means describe the scored subset; "
            "incomplete panels have no full-panel mean or ranking. "
            "Intervals use independent worlds; absent intervals indicate withheld inference."
            " The evidence classification is supplied by the operator; verify it against the linked provenance."
        )
        headings = (
            "model",
            "track",
            "mean_loss",
            "interval_low",
            "interval_high",
            "standard_error",
            "worlds",
            "attempts",
            "scored_attempts",
            "observed_mean_loss",
            "scored_worlds",
            "scored_cell_refs",
        )
        rows = [
            (
                row.model_id,
                row.target_kind,
                str(row.mean_loss),
                str(row.interval_low),
                str(row.interval_high),
                str(row.standard_error),
                str(row.independent_worlds),
                str(row.scheduled_attempts),
                str(row.scored_attempts),
                str(row.observed_mean_loss),
                str(row.scored_worlds),
                ", ".join(row.scored_cell_refs),
            )
            for row in report.loss_rankings
        ]
        plot_rows = [
            (
                f"{row.model_id} / {row.target_kind}",
                row.mean_loss,
                row.interval_low,
                row.interval_high,
                row.independent_worlds,
                row.scheduled_attempts,
            )
            for row in report.loss_rankings
            if row.mean_loss is not None
        ]
    elif isinstance(report, VerificationReportV1) and report.operation == "analyse-trialdev":
        title = f"Scheduled development results — {evidence_kind or 'unclassified'}"
        scope = (
            f"Verification status: {report.status}. Independent arithmetic inspects retained execution custody; "
            "it does not perform fresh execution or assess current authorization. "
            "Observational loss and expected complete-programme value are separate metrics. "
            "Means weight worlds equally within each profile; scheduled failures remain included. "
            "Evaluator errors withhold affected means. No confidence interval is asserted. "
            "The operator-supplied evidence classification requires provenance review."
        )
        ranking = TrialDevRunRankingV1.model_validate(report.checks[0].observed)
        headings = (
            "stream",
            "data_view",
            "asset_count",
            "clinical_profile",
            "scenario_profile",
            "metric",
            "mean",
            "worlds",
            "attempts",
            "scored_tasks",
            "observed_mean",
            "scored_worlds",
            "unscored_tasks",
            "model_noncompletion",
            "initial_targets_scored/scheduled",
            "initial_target_status_counts",
            "evaluator_errors",
        )
        rows = [
            (
                row.stream,
                row.data_view,
                str(row.asset_count),
                row.clinical_profile_id,
                row.scenario_profile_id,
                row.metric,
                str(row.mean),
                str(row.scheduled_worlds),
                str(row.scheduled_tasks),
                str(row.scored_tasks),
                str(row.observed_mean),
                str(row.scored_worlds),
                str(row.unscored_task_counts),
                str(row.model_noncompletion),
                (
                    "not applicable"
                    if row.target_status_counts is None
                    else f"{row.target_status_counts.get('reported', 0)}/{sum(row.target_status_counts.values())}"
                ),
                str(row.target_status_counts),
                str(row.evaluator_errors),
            )
            for row in ranking.profile_means
        ]
    elif isinstance(report, VerificationFailureReportV1):
        title = "Verification failed"
        scope = "No successful verification or official score is established by this failed operation."
        headings = ("operation", "failure", "detail")
        rows = [(report.operation, report.failure_code, report.detail)]
    else:
        fixture = isinstance(report, PracticeVerificationReportV1)
        title = (
            "Disclosed arithmetic controls"
            if fixture
            else f"Verification of supplied artifacts — {evidence_kind or 'unclassified'}"
        )
        scope = (
            "Disclosed installation fixtures, not benchmark items or model results. "
            "Recovery controls show the penalty for uninformative width and confident error; "
            "they do not estimate coverage or empirical model performance."
            if fixture
            else f"Operation: {report.operation}; status: {report.status}. "
            "Checks apply only to the supplied, bound artifacts. Authorization is not assessed. "
            "Integrity, reference checks, score agreement and fresh execution are separate claims. "
            "This report does not perform fresh code execution or certify clinical validity."
        )
        if not fixture and report.operation == "score":
            scope += (
                " Score checks retain scientific failures and unevaluable diagnostics; a failed check alone "
                "does not establish disagreement with native grading. Campaign analysis checks that agreement "
                "against the retained numerical grades."
            )
        headings = ("scope", "check", "status", "observed", "expected")
        rows = [
            (
                check.scope_id,
                check.check_id,
                check.status,
                json.dumps(check.observed, sort_keys=True),
                json.dumps(check.expected, sort_keys=True),
            )
            for check in report.checks
        ]
        if fixture:
            plot_rows = [
                (check.scope_id.removeprefix("loss."), check.observed, None, None, None, None)
                for check in report.checks
                if check.scope_id.startswith("loss.") and isinstance(check.observed, float)
            ]
    if isinstance(report, VerificationReportV1) and report.development:
        title = f"Unqualified rehearsal — {title}"
        scope = "Development inputs: these results do not establish benchmark qualification. " + scope
    with (directory / "checks.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((*headings, "scorer"))
        writer.writerows((*row, scorer_json) for row in rows)
    contrasts_link = ""
    if isinstance(report, TrialEvalCampaignAnalysisV1) and report.between_model_contrasts:
        # Paired between-model differences with their standard errors and minimal detectable differences.
        with (directory / "model_contrasts.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scorer",
                    "minuend_model",
                    "subtrahend_model",
                    "context",
                    "metric",
                    "status",
                    "estimate",
                    "interval_low",
                    "interval_high",
                    "standard_error",
                    "minimal_detectable_difference",
                    "p_value",
                    "holm_adjusted_p_value",
                    "observed_pairs",
                    "scheduled_pairs",
                )
            )
            writer.writerows(
                (
                    row.scorer.model_dump_json(),
                    row.minuend_model_id,
                    row.subtrahend_model_id,
                    row.context,
                    row.metric.metric_id,
                    row.status,
                    row.estimate,
                    row.interval_low,
                    row.interval_high,
                    row.standard_error,
                    row.minimal_detectable_difference,
                    row.p_value,
                    row.adjusted_p_value,
                    row.observed_pairs,
                    row.scheduled_pairs,
                )
                for row in report.between_model_contrasts
            )
        contrasts_link = (
            '; <a href="model_contrasts.csv">paired model contrasts, including C5 decision headlines, '
            "standard errors and minimal detectable differences</a>"
        )
    figure_html = ""
    if plot_rows:
        with (directory / "loss.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                (
                    "scorer",
                    "label",
                    "mean_loss",
                    "interval_low",
                    "interval_high",
                    "worlds",
                    "attempts",
                )
            )
            writer.writerows((scorer_json, *row) for row in plot_rows)
        figure = Figure(figsize=(10, max(3.5, 1.4 + 0.9 * len(plot_rows))), layout="constrained")
        axes = figure.subplots()
        for index, (_label, value, low, high, _worlds, _attempts) in enumerate(plot_rows):
            axes.plot(value, index, "o", color="#176A80")
            if low is not None and high is not None:
                axes.hlines(index, low, high, color="#176A80", linewidth=2)
        labels = [
            textwrap.fill(
                (label if worlds is None else f"{label} (worlds={worlds}, attempts={attempts})").replace("_", " "),
                width=48,
            )
            for label, _, _, _, worlds, attempts in plot_rows
        ]
        axes.set_yticks(range(len(labels)), labels)
        axes.invert_yaxis()
        axes.set(xlabel="Mean normalised loss (lower is better)")
        axes.set_xlim(left=0)
        figure.suptitle(title.replace("_", " "))
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="x", alpha=0.2)
        with rc_context({"svg.hashsalt": digest}):
            figure.savefig(directory / "loss.svg", metadata={"Date": None, "Description": scorer_json})
        figure_html = (
            '<figure><img src="loss.svg" alt="Loss values with available uncertainty intervals; '
            'exact values and denominators are in loss.csv">'
            '<figcaption>See <a href="loss.csv">the accessible figure table</a>. '
            "No interval is inferred for a disclosed control or an unqualified stratum.</figcaption></figure>"
        )
    header = "".join(f"<th>{html.escape(name.replace('_', ' '))}</th>" for name in headings)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="trialagentbench-scorer" content="{html.escape(scorer_json)}">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:100rem}"
        ".table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%}"
        "th,td{padding:.5rem;border-bottom:1px solid #bbb;text-align:left;min-width:5rem;max-width:32rem;"
        "overflow-wrap:break-word}img{max-width:100%}code{overflow-wrap:anywhere}</style>"
        f"<main><h1>{html.escape(title)}</h1><p>{html.escape(scope)}</p>"
        f"<p>Result SHA-256: <code>{digest}</code>. "
        '<a href="report.json">Full result and provenance</a>; '
        f'<a href="checks.csv">result table</a>{contrasts_link}. Keep exports under the input access restrictions.</p>'
        f'{figure_html}<div class="table-scroll" role="region" aria-label="Result table" tabindex="0">'
        f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div></main></html>"
    )
    (directory / "index.html").write_text(document, encoding="utf-8")
