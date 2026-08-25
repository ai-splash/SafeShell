"""
Shared LLM JSON parsing/repair helpers.

Local LLMs asked to "respond with only JSON" still frequently wrap the
reply in markdown fences, prepend a `<think>` block, add stray prose, or
make small JSON formatting mistakes (trailing commas, Python literals).
This module turns a raw model reply into a Python dict as reliably as
possible without ever silently fabricating data - if nothing usable can be
recovered, `parse_json_object` raises and the caller decides the fallback.

Extracted out of `ai_assistant.py` so both the original explain-only
pipeline and the newer tool-grounded ops pipeline (`ops_assistant.py`)
share one implementation instead of two copies drifting apart.
"""

from __future__ import annotations

import json
import re

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` even when told not to."""
    return _JSON_FENCE_RE.sub("", text).strip()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning preambles some models emit even
    when 'think' is disabled, so they never leak into the JSON we parse."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def extract_json_object(text: str) -> str | None:
    """Find the first balanced {...} object in text.

    Handles cases where the model surrounds the JSON with extra prose
    ("Sure, here's the JSON: {...} Let me know if...") by scanning for the
    first '{' and tracking brace depth (ignoring braces inside quoted
    strings) until it finds the matching closing '}'. Returns None if no
    balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def repair_json(text: str) -> str:
    """Best-effort fix for minor JSON formatting mistakes before giving up.

    Targets the mistakes small/local LLMs actually make: trailing commas
    before a closing bracket, Python-style None/True/False literals instead
    of JSON's null/true/false, and stray control characters that break
    string parsing. Deliberately conservative - it only rewrites patterns
    that are unambiguous, so it never turns valid JSON into something else.
    """
    repaired = _CONTROL_CHAR_RE.sub("", text)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def parse_json_object(raw: str) -> dict:
    """Turn a raw LLM reply into a JSON dict, tolerating common formatting
    slop: markdown fences, <think> preambles, surrounding prose, trailing
    commas, and Python-style literals. Raises ValueError if, even after all
    of that, no valid JSON object can be recovered.
    """
    cleaned = strip_think_blocks(strip_code_fences(raw))
    candidate = extract_json_object(cleaned) or cleaned

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        data = json.loads(repair_json(candidate))  # let this raise if it still fails

    if not isinstance(data, dict):
        raise ValueError("Top-level JSON was not an object")
    return data
