"""Markdown letters returned by the programme tools: status synopsis, readout and decision."""

from __future__ import annotations

from trialagentbench.trialdev.actions import TrialDevProgrammeStatus, TrialDevStartStudy
from trialagentbench.trialdev.materialization import TrialDevReadoutManifest

_PHASE = {"early_safety": "early safety", "exploration": "exploration", "confirmation": "confirmation"}
_ENDPOINT = {"clinical_outcome": "clinical endpoint", "safety_risk": "safety-event risk"}
CONTROL_REFERENCE_NOTE = (
    "Treatment contrasts are candidate (or active arm) minus control, in the stated natural units."
)
"""The reference treatment and direction of the requested contrasts."""


def _usd(value: float) -> str:
    return f"${value / 1e6:,.1f}M"


def programme_synopsis(status: TrialDevProgrammeStatus) -> str:
    """Render the programme clock, money and every study in a few lines."""
    lines = [
        f"Programme day {status.programme_day:,.0f} of {status.horizon_days:,}. Budget {_usd(status.budget_usd)}: "
        f"spent {_usd(status.spent_usd)}, committed to follow-up {_usd(status.committed_followup_usd)}, "
        f"available {_usd(status.available_usd)}."
    ]
    if status.studies:
        lines += ["", "| Study | Phase | Arms: enrolled / cap | Recruitment | Readouts done | Next look deadline |"]
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for study in status.studies:
            arms = "; ".join(f"{arm.arm_id} {arm.enrolled}/{arm.enrollment_cap}" for arm in study.arms)
            recruitment = (
                "closed"
                if study.closed
                else (
                    "paused"
                    if study.recruitment_paused
                    else "complete" if all(arm.recruitment_closed for arm in study.arms) else "open"
                )
            )
            looks = f"{study.completed_looks} of {study.completed_looks + len(study.remaining_looks)}"
            deadline = f"day {study.remaining_looks[0].deadline_day:,.0f}" if study.remaining_looks else "none"
            lines.append(
                f"| {study.study_id} | {_PHASE[study.phase]} | {arms} | {recruitment} | {looks} | {deadline} |"
            )
    return "\n".join(lines)


def design_letter(design: TrialDevStartStudy, status: TrialDevProgrammeStatus, *, preview: bool) -> str:
    """Render a started (or previewed) study's resolved design and the programme after it."""
    title = f"## {design.study_id} " + (
        "preview (not started)" if preview else f"started on day {status.programme_day:,.0f}"
    )
    arms = "; ".join(
        f"{arm.arm_id} ({arm.asset_id}{'' if arm.dose is None else ' ' + arm.dose.value}) {arm.participant_count}"
        for arm in design.arms
    )
    looks = "; ".join(
        f"by day {look.deadline_day:,.0f}"
        + ("" if look.completed_participants is None else f" or at {look.completed_participants} completed")
        for look in design.looks
    )
    lines = [
        title,
        "",
        f"- Phase: {_PHASE[design.phase]}; {design.sites} sites; follow-up {design.follow_up_days} days",
        f"- Arms and enrollment caps: {arms}",
        f"- Assessment days: {', '.join(map(str, design.assessment_days))}; "
        f"safety contact days: {', '.join(map(str, design.surveillance_days))}",
        f"- Readouts: {looks}",
        "",
        ("Programme if started now:" if preview else "Programme now:"),
        programme_synopsis(status),
    ]
    if preview:
        lines += ["", "Call start_study with the same arguments (without preview) to start it."]
    return "\n".join(lines)


def readout_letter(
    manifest: TrialDevReadoutManifest,
    *,
    folder: str,
    look: int,
    status: TrialDevProgrammeStatus,
    previous_capture_error: str | None,
) -> str:
    """Render one readout: where the files are, one time origin, and the results now requested."""
    arms = ", ".join(arm.arm_id for arm in manifest.arms)
    rows = "\n".join(
        f"| `{target.target_id}` | {target.arm_id} minus control, {_ENDPOINT[target.endpoint]} | {target.unit} |"
        for target in manifest.targets
    )
    lines = [
        f"## {manifest.study_id} readout {look} (programme day {manifest.readout_day:,.0f})",
        "",
        f"Files: `{folder}/` — study.json (design, targets), data_dictionary.csv, and arms/<arm>/*.parquet for "
        f"arms {arms}. These files are final; a later look is released in a new folder. Event and record "
        "dates and enrollment_day are programme days; subtract enrollment_day for time since enrolment. "
        "Assessment schedules and follow_up_days are days since enrolment. Optional consent_time in "
        "baseline.parquet is also days since enrolment; consult data_dictionary.csv for each field's clock.",
        "",
        f"Results requested for this readout, in this order (interface/targets.csv): estimates at follow-up day "
        f"{manifest.follow_up_days} for the whole programme population, with 95% confidence intervals.",
        "",
        "| Target | Measure | Unit |",
        "| --- | --- | --- |",
        rows,
        "",
        CONTROL_REFERENCE_NOTE,
        "",
        "Update your saved analysis for these targets. Your next readout request or final decision runs it and "
        "keeps its output as your report for this readout.",
    ]
    if previous_capture_error:
        lines += ["", f"Note: the analysis kept for the previous cutoff was incomplete: {previous_capture_error}"]
    return "\n".join([*lines, "", programme_synopsis(status)])


def decision_letter(status: TrialDevProgrammeStatus, *, previous_capture_error: str | None) -> str:
    """Render the terminal development decision."""
    terminal = status.terminal
    if terminal is None:
        raise ValueError("A decision letter requires a terminal programme decision")
    if terminal.decision == "stop":
        headline = f"## Development stopped on programme day {terminal.programme_day:,.0f}"
    else:
        assert terminal.dose is not None
        headline = (
            f"## {terminal.asset_id} ({terminal.dose.value}) advanced to benchmark submission on programme day "
            f"{terminal.programme_day:,.0f}"
        )
    lines = [
        headline,
        "",
        programme_synopsis(status),
        "",
        "The programme is complete; no further actions are possible.",
    ]
    if previous_capture_error:
        lines[1:1] = [f"Note: the analysis kept at this cutoff was incomplete: {previous_capture_error}", ""]
    return "\n".join(lines)


def status_letter(status: TrialDevProgrammeStatus, *, change: str) -> str:
    """Render the outcome of a study change or a wait."""
    return f"## {change}\n\n{programme_synopsis(status)}"


__all__ = [
    "decision_letter",
    "design_letter",
    "programme_synopsis",
    "readout_letter",
    "status_letter",
]
