"""Public TrialAgentBench command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

CommandHandler = Callable[[Sequence[str] | None], int]


def demo_run(argv: Sequence[str] | None) -> int:
    """Run the disclosed provider-free native teaching walkthrough."""
    from trialagentbench.tools.demo import main

    return main(argv)


def campaign_action(action: str, argv: Sequence[str] | None) -> int:
    """Operate a saved many-model native campaign."""
    from trialagentbench.tools.campaign import main

    return main([action, *(argv or ())])


def run_trialeval(argv: Sequence[str] | None) -> int:
    """Run TrialEval."""
    from trialagentbench.tools.run.trialeval import main

    return main(argv)


def run_trialdev(argv: Sequence[str] | None) -> int:
    """Run TrialDev."""
    from trialagentbench.tools.run.canonical_trialdev import main

    return main(argv)


def grade_trialeval(argv: Sequence[str] | None) -> int:
    """Grade TrialEval."""
    from trialagentbench.tools.grade.grade_trialeval import grade_trialeval_run

    return grade_trialeval_run(list(argv) if argv is not None else None)


def grade_trialdev(argv: Sequence[str] | None) -> int:
    """Grade TrialDev."""
    from trialagentbench.tools.grade.canonical_trialdev import main

    return main(argv)


def analyse_trace(argv: Sequence[str] | None) -> int:
    """Analyse evaluation traces."""
    from trialagentbench.tools.build.build_trace_analysis_bundle import main

    return main(list(argv) if argv is not None else None)


def analyse_calibration(argv: Sequence[str] | None) -> int:
    """Analyse calibration runs."""
    from trialagentbench.tools.analyse_calibration import main

    return main(argv)


def validate_submission(argv: Sequence[str] | None) -> int:
    """Validate a TrialEval submission."""
    from trialagentbench.tools.validate.validate_submission import main

    return main(argv)


def export_results(argv: Sequence[str] | None) -> int:
    """Export benchmark results."""
    from trialagentbench.tools.export_results import main

    return main(argv)


@dataclass(frozen=True)
class _Command:
    handler: CommandHandler
    description: str

    def invoke(self, argv: Sequence[str]) -> int:
        result = self.handler(list(argv))
        return int(result or 0)


_COMMANDS: dict[str, dict[str, _Command]] = {
    "demo": {
        "run": _Command(demo_run, "Run a scripted clinical research demonstration without provider credentials.")
    },
    "campaign": {
        action: _Command(partial(campaign_action, action), f"{action.capitalize()} a native benchmark campaign.")
        for action in ("preview", "prepare", "run", "status", "resume", "finish", "regrade")
    }
    | {
        "retry": _Command(
            partial(campaign_action, "retry"),
            "Retry once every TrialEval primary that ended in an infrastructure failure.",
        )
    },
    "run": {
        "trialeval": _Command(run_trialeval, "Run TrialEval."),
        "trialdev": _Command(run_trialdev, "Run TrialDev."),
    },
    "grade": {
        "trialeval": _Command(
            grade_trialeval,
            "Grade a canonical TrialEval run.",
        ),
        "trialdev": _Command(
            grade_trialdev,
            "Grade a canonical TrialDev run.",
        ),
    },
    "analyse": {
        "trace": _Command(
            analyse_trace,
            "Build trace analysis from user-owned runs.",
        ),
        "calibration": _Command(
            analyse_calibration,
            "Summarize scheduled runs, grades, failures, and resource use.",
        ),
    },
    "verify": {
        "submission": _Command(
            validate_submission,
            "Validate an executable TrialEval numerical result without evaluator answers.",
        ),
    },
    "export": {
        "results": _Command(
            export_results,
            "Export a deterministic, checksummed result bundle.",
        ),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench",
        description="Run, grade, analyse, verify, and export TrialAgentBench results.",
    )
    workflows = parser.add_subparsers(dest="workflow", required=True)
    for workflow, commands in _COMMANDS.items():
        workflow_parser = workflows.add_parser(workflow)
        actions = workflow_parser.add_subparsers(dest="action", required=True)
        for action, command in commands.items():
            actions.add_parser(
                action,
                description=command.description,
                help=command.description,
                add_help=False,
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one public TrialAgentBench workflow.

    A provider transport failure stops the command with status 2 and only the
    harness's own message: the SDK error it wraps can echo request headers or
    response bodies, so its traceback is not printed.
    """
    from trialagentbench.ports.llm_provider import ProviderRequestError

    tokens = list(argv) if argv is not None else sys.argv[1:]
    parser = _parser()
    args, leaf_argv = parser.parse_known_args(tokens)
    command = _COMMANDS[args.workflow][args.action]
    try:
        return command.invoke(leaf_argv)
    except ProviderRequestError as error:
        print(f"Provider request failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
