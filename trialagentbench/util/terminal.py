"""Render untrusted text without terminal or Unicode direction controls."""

from __future__ import annotations

import unicodedata


def terminal_text(text: str) -> str:
    """Strip control, format and surrogate characters, preserving newlines and tabs."""

    return "".join(char for char in text if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
