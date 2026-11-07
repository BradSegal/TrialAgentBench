"""Remove authentication material at provider diagnostic boundaries."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from trialagentbench.ports.llm_provider import JsonValue

_TOKEN = re.compile(r"(?i)\bBearer\s+[^\s\"'<>;,}\]]+|\bsk-[A-Za-z0-9_*.…-]+")
_MASKED_KEY = re.compile(r"(?i)(?<!\w)[a-z0-9_-]{4,}(?:\*{3,}|\.{3,}|…)[a-z0-9_-]{4,}(?!\w)")
_MASKED = re.compile(r"(?i)\b(?:api[_ -]?key|authorization|token)\s*[:=]\s*[\"']?[^\s\"'<>;,}\]]+")


def redact_provider_text(text: str, *, credentials: tuple[str, ...] = (), match_patterns: bool = True) -> str:
    """Redact exact configured keys and recognizable tokens before any truncation."""

    keys = set(credentials)
    # Include custom provider key variables without inspecting unrelated environment values.
    for name in os.environ:
        if name.endswith("_API_KEY") or name == "open_router_key":
            value = os.environ[name]
            if value:
                keys.add(value)
    for key in sorted(keys, key=len, reverse=True):
        if key:
            text = text.replace(key, "[REDACTED]")
    if not match_patterns:
        return text
    return _MASKED.sub("[REDACTED]", _MASKED_KEY.sub("[REDACTED]", _TOKEN.sub("[REDACTED]", text)))


def redact_provider_value(
    value: JsonValue, *, credentials: tuple[str, ...] = (), match_patterns: bool = True
) -> JsonValue:
    """Redact strings recursively in retained provider JSON, including object keys."""

    if isinstance(value, str):
        return redact_provider_text(value, credentials=credentials, match_patterns=match_patterns)
    if isinstance(value, list):
        return [redact_provider_value(item, credentials=credentials, match_patterns=match_patterns) for item in value]
    if isinstance(value, Mapping):
        return {
            redact_provider_text(key, credentials=credentials, match_patterns=match_patterns): redact_provider_value(
                item, credentials=credentials, match_patterns=match_patterns
            )
            for key, item in value.items()
        }
    return value


def redact_provider_exception(error: BaseException, *, credentials: tuple[str, ...] = ()) -> None:
    """Sanitize SDK exception chains before telemetry or a traceback can render them."""

    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.args = tuple(
            redact_provider_text(value, credentials=credentials) if isinstance(value, str) else value
            for value in current.args
        )
        # SDK errors render .message instead of .args on some versions.
        message = getattr(current, "message", None)
        if isinstance(message, str):
            current.message = redact_provider_text(message, credentials=credentials)  # type: ignore[attr-defined]
        notes = getattr(current, "__notes__", None)
        if isinstance(notes, list):
            current.__notes__ = [redact_provider_text(note, credentials=credentials) for note in notes]
        raw_response = getattr(current, "raw_response", None)
        if isinstance(raw_response, dict):
            current.raw_response = redact_provider_value(raw_response, credentials=credentials)  # type: ignore[attr-defined]
        body = getattr(current, "body", None)
        if isinstance(body, (dict, list, str)):
            current.body = redact_provider_value(body, credentials=credentials)  # type: ignore[attr-defined]
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
