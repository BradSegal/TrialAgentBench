"""Run the native demonstration with an external provider implementation.

This complete, credential-free adapter implements the same protocol that a real
agent or model client implements. Replace generate_turn's body with your client
while preserving the response/tool-call and accurate usage contracts. The native
harness owns evidence, execution, acquisition, reporting, checkpoints and grades.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from trialagentbench.ports.llm_provider import JsonObject, LLMResponse
from trialagentbench.tools.demo import ScriptedDemoProvider, run_demo


class ExampleProvider:
    """An external adapter that delegates to the disclosed local teaching policy."""

    model = ScriptedDemoProvider.model
    telemetry_route = "local-scripted"

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
        """Read participant messages and return ordinary public tool requests."""
        return ScriptedDemoProvider().generate_turn(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            tool_choice=tool_choice,
        )


def main() -> None:
    """Run the same native walkthrough with this independently defined adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-image")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_demo(output=args.output, provider=ExampleProvider(), executor_image=args.executor_image, resume=args.resume)


if __name__ == "__main__":
    main()
