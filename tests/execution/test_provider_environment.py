from __future__ import annotations

import os
from pathlib import Path

import pytest

from trialagentbench.util.provider_environment import load_provider_dotenv


def test_dotenv_defaults_to_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone commands load the opt-in credential file beside the user."""

    (tmp_path / ".env").write_text("open_router_key=working-directory-secret\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("open_router_key", raising=False)

    load_provider_dotenv()

    assert os.environ["OPENROUTER_API_KEY"] == "working-directory-secret"
    assert "open_router_key" not in os.environ


def test_dotenv_maps_lowercase_openrouter_key_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("open_router_key=test-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("open_router_key", raising=False)

    load_provider_dotenv(dotenv_path=dotenv_path)

    assert os.environ["OPENROUTER_API_KEY"] == "test-secret"
    assert "open_router_key" not in os.environ


def test_dotenv_rejects_conflicting_openrouter_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("open_router_key=lower-secret\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "upper-secret")
    monkeypatch.delenv("open_router_key", raising=False)

    with pytest.raises(ValueError, match="conflicting"):
        load_provider_dotenv(dotenv_path=dotenv_path)

    assert os.environ["OPENROUTER_API_KEY"] == "upper-secret"
    assert "open_router_key" not in os.environ


def test_dotenv_preserves_standard_uppercase_openrouter_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("OPENROUTER_API_KEY=standard-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("open_router_key", raising=False)

    load_provider_dotenv(dotenv_path=dotenv_path)

    assert os.environ["OPENROUTER_API_KEY"] == "standard-secret"
    assert "open_router_key" not in os.environ
