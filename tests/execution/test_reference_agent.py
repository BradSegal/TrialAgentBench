"""The scripted reference agent refuses tools the runtime did not offer."""

from __future__ import annotations

import pytest
from reference_agent.agent import ReferenceAgent


def test_reference_agent_requests_only_offered_tools() -> None:
    """The agent refuses to call a tool the runtime did not offer."""
    agent = ReferenceAgent()
    messages = [
        {"role": "system", "content": "Data files: data/raw/a.parquet.\n"},
        {"role": "user", "content": "q"},
    ]
    with pytest.raises(ValueError, match="unavailable tool"):
        agent.generate_turn(
            messages, [{"type": "function", "function": {"name": "submit_response"}}], temperature=0, max_tokens=1
        )
