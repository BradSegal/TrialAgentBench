"""Run your own agent on one benchmark task, then reprice its finished programme with the verifier.

The provider below is a local script, so this makes no model calls and needs no key. To connect an agent, replace
the body of ``generate_turn`` with a call to your client, returning the same ``LLMResponse``. The harness owns the
evidence, the study actions, the analysis sandbox and the grade. It reads a downloaded release in ``benchmark/``.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
from trialagentbench_verifier.executor import resolve_executor_environment
from trialagentbench_verifier.resource_accounting import reconstruct_trialdev_terminal_value

from trialagentbench.adapters.offline_control import OfflineControlProvider
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.ports.llm_provider import JsonObject, LLMResponse
from trialagentbench.trialdev.release import load_trialdev_evaluator_release
from trialagentbench.trialdev.run import run_trialdev_task

# An atopic dermatitis programme from the headline panel, at its US$120M budget.
TASK = "td-0759a15209f1"


class MyAgent:
    """Everything the harness asks of an agent: a name, a route and one method."""

    model = "smoke-zero-python"
    telemetry_route = "offline_control"

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Read the participant messages and offered tools, and return the next tool calls."""
        return OfflineControlProvider(self.model).generate_turn(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            tool_choice=tool_choice,
        )


def main() -> None:
    """Run the task, then recompute its development value from the retained ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the run")
    parser.add_argument("--release", type=Path, default=Path("benchmark/full"))
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()
    catalog = args.release / "trialdev-evaluator.assets.json"
    digests = {name: digest for digest, name in map(str.split, (args.release / "SHA256SUMS").read_text().splitlines())}
    executor = resolve_executor_environment(limits=RELEASE_EXECUTION_LIMITS_V1)
    with materialized_release_archive(
        catalog, expected_sha256=digests[catalog.name], task_ids=(args.task,)
    ) as materialized:
        release = load_trialdev_evaluator_release(materialized.root)
        run = run_trialdev_task(
            release=release,
            task_id=args.task,
            provider=MyAgent(),
            output_root=args.output,
            max_turns_per_step=100,
            max_tokens=4000,
            max_context_characters=400_000,
            watchdog_seconds=900,
            tool_choice="auto",
            executor_image=executor.image_id,
            executor_limits=executor.limits,
        )
        task = next(item for item in release.participant.tasks if item.task_id == args.task)
        value = reconstruct_trialdev_terminal_value(
            ledger_path=args.output / args.task / "private/budget.json",
            configuration_path=materialized.root / "evaluation/tasks" / args.task / "authored_programme.json",
            pool_path=materialized.root / "evaluation/worlds" / task.case_id / "pool/manifest.json",
        )
    print(f"{args.task}: {run.execution_status}; development value US${value.net_value_usd / 1e6:.1f}M")


if __name__ == "__main__":
    main()
