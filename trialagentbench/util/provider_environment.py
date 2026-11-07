"""Provider credential loading for explicitly requested live runs."""

from __future__ import annotations

import os
from pathlib import Path


def load_provider_dotenv(*, dotenv_path: Path | None = None) -> None:
    """Load provider credentials from an explicit or working-directory file."""

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise RuntimeError("python-dotenv is required for --dotenv") from exc

    resolved_path = dotenv_path if dotenv_path is not None else Path.cwd() / ".env"
    load_dotenv(dotenv_path=resolved_path)
    lowercase_key = os.environ.pop("open_router_key", None)
    if lowercase_key is None:
        return
    if not lowercase_key:
        raise ValueError("open_router_key must not be empty.")
    uppercase_key = os.environ.get("OPENROUTER_API_KEY")
    if uppercase_key is not None and uppercase_key != lowercase_key:
        raise ValueError("OPENROUTER_API_KEY and open_router_key contain conflicting values.")
    os.environ["OPENROUTER_API_KEY"] = lowercase_key


__all__ = ["load_provider_dotenv"]
