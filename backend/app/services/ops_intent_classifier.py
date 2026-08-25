"""
Ops intent classifier for the real-time Linux Operations Copilot.

This is a second, narrower classifier that sits in front of the existing
`intent_classifier.py`. Where that one decides a broad *topic* (performance,
network, logs, ...) so the old psutil-based context builder can gather
loosely-related data, this one decides whether the question is one of the
specific, high-precision "run this exact tool and answer from live data"
questions this upgrade targets - e.g. "show top 5 CPU consuming processes"
-> run the CPU tool, not just "this is a performance question".

Deliberately rule-based and fully offline for the same reasons as the
existing classifier: instant, free, and this routing step doesn't need an
LLM call to be reliable for a fixed, well-known set of ops questions.

`classify_ops_intent()` returns `None` when the question doesn't clearly
match one of the whitelisted tools, so callers (see `ai_assistant.py`) can
fall back to the existing general-purpose assistant pipeline unchanged.

Service-level intelligence (added): in addition to the original
whole-fleet tools (list running/failed services), this classifier now also
recognizes questions that name (or clearly imply) ONE specific service -
"Is nginx running?", "Why is PostgreSQL down?", "Check the ssh service" -
and extracts a best-effort service name hint (`service_hint`) for those.
The hint is just free text pulled out of the question; it is NOT trusted as
a real unit name anywhere - `ops_assistant.py` resolves it against the
live, actual list of systemd units before anything is executed, and
`tool_executor.normalize_unit_name` validates/sanitizes it before it can
ever reach a command argv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.tool_executor import ToolName


@dataclass
class OpsIntentResult:
    tool: ToolName
    matched_patterns: list[str]
    confidence: float
    # Populated only for single-service questions ("is nginx running?").
    # Free-text hint extracted from the question - NOT a trusted unit name.
    service_hint: str | None = None
    # Whether this question is asking "why"/troubleshooting a specific
    # service, in which case the recent journal for that unit should be
    # pulled alongside its status.
    needs_logs: bool = False


# --------------------------------------------------------------------------
# Single-service detection - checked BEFORE the generic fleet-wide patterns
# below, so "is nginx running?" is answered about nginx specifically rather
# than being swallowed by the generic "running services" pattern.
#
# Each pattern has one capturing group named `svc`. Common filler/keyword
# words that could otherwise be mistaken for a service name (e.g. "is ANY
# service down") are filtered out via `_SERVICE_HINT_STOPWORDS`.
# --------------------------------------------------------------------------

_SERVICE_HINT_STOPWORDS = {
    "service", "services", "unit", "units", "running", "active", "status",
    "failed", "failing", "fail", "logs", "log", "errors", "error", "ports",
    "port", "network", "cpu", "memory", "disk", "the", "a", "an", "my",
    "any", "some", "currently", "down", "up", "enabled", "disabled",
    "healthy", "unhealthy", "ok", "there", "important", "system", "systemd",
    "server", "recent", "latest",
}

# Systemd unit names may contain letters, digits, and `_.-@:\`.
_SVC_TOKEN = r"[A-Za-z0-9_.\-]+"

# (regex, resulting tool, needs_logs)
_SERVICE_HINT_PATTERNS: list[tuple[str, ToolName, bool]] = [
    (rf"\brecent\s+logs?\s+for\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_LOGS, True),
    (rf"\bshow\b.*\blogs?\s+for\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_LOGS, True),
    (rf"\blogs?\s+for\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_LOGS, True),
    (
        rf"\bwhy\s+(?:is|did|was|does)\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\s+"
        rf"(?:not\s+running|down|fail(?:ing|ed)?|crash(?:ing|ed)?)\b",
        ToolName.SERVICE_STATUS,
        True,
    ),
    (rf"\bwhy\s+(?:did\s+)?(?P<svc>{_SVC_TOKEN})\s+fail(?:ed)?\b", ToolName.SERVICE_STATUS, True),
    (rf"\bwhat\s+happened\s+to\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_STATUS, True),
    (rf"\bis\s+there\s+an?\s+error\s+with\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_STATUS, True),
    (
        rf"\bwhat\s+should\s+i\s+check\s+if\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\s+is\s+down\b",
        ToolName.SERVICE_STATUS,
        True,
    ),
    (
        rf"\bis\s+(?:my\s+)?(?P<svc>{_SVC_TOKEN})\s+service\s+"
        rf"(?:healthy|running|active|up|ok)\b",
        ToolName.SERVICE_STATUS,
        False,
    ),
    (
        rf"\bis\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\s+service\s+"
        rf"(?:active|running|up|healthy|ok)\b",
        ToolName.SERVICE_STATUS,
        False,
    ),
    (rf"\bis\s+(?P<svc>{_SVC_TOKEN})\.service\s+(?:active|running)\b", ToolName.SERVICE_STATUS, False),
    (
        rf"\bis\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\s+"
        rf"(?:currently\s+)?(?:running|active|up|healthy|ok|alive)\b",
        ToolName.SERVICE_STATUS,
        False,
    ),
    (rf"\bis\s+(?P<svc>{_SVC_TOKEN})\s+enabled\b", ToolName.SERVICE_STATUS, False),
    (rf"\bis\s+(?P<svc>{_SVC_TOKEN})\s+disabled\b", ToolName.SERVICE_STATUS, False),
    (
        rf"\bwhat(?:'s|\s+is)\s+the\s+status\s+of\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b",
        ToolName.SERVICE_STATUS,
        False,
    ),
    (rf"\bstatus\s+of\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_STATUS, False),
    (rf"\bcheck\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\s+service\b", ToolName.SERVICE_STATUS, False),
    (rf"\bcheck\s+(?:the\s+)?(?P<svc>{_SVC_TOKEN})\b", ToolName.SERVICE_STATUS, False),
    (rf"\b(?P<svc>{_SVC_TOKEN})\.service\b", ToolName.SERVICE_STATUS, False),
]


def _extract_service_hint(text: str) -> tuple[str, ToolName, bool] | None:
    """Try each single-service pattern in order; return the first plausible
    (service_hint, tool, needs_logs) match, or None if nothing plausible
    matched. "Plausible" excludes generic/filler words via the stopword
    list, so e.g. "is any service down" doesn't get treated as a service
    literally named "any".
    """
    for pattern, tool, needs_logs in _SERVICE_HINT_PATTERNS:
        match = re.search(pattern, text)
        if not match or "svc" not in match.groupdict():
            continue
        svc = (match.group("svc") or "").strip(" .:-")
        if not svc or svc.lower() in _SERVICE_HINT_STOPWORDS:
            continue
        return svc, tool, needs_logs
    return None


# --------------------------------------------------------------------------
# Fleet-wide / generic patterns - checked only when no specific service was
# detected above. Ordered so more specific patterns are checked before more
# generic ones, e.g. "failed services" must win over the generic "services"
# pattern, and "recent errors" must win over the generic "recent logs"
# pattern, and "all services" must win over "running services".
# --------------------------------------------------------------------------
_OPS_PATTERNS: list[tuple[ToolName, list[str]]] = [
    (
        ToolName.SERVICES_FAILED,
        [
            r"\bfailed\s+services?\b",
            r"\bservices?\s+(that\s+)?(failed|have\s+failed|are\s+down)\b",
            r"\bservice\s+failures?\b",
            r"\bwhich\s+services?\s+failed\b",
            r"\bany\s+failed\s+units?\b",
            r"\bis\s+any\s+service\s+.*\bdown\b",
            r"\bany\s+service\s+.*\b(down|failing)\b",
            r"\bservices?\s+(are\s+)?failing\b",
            r"\bimportant\s+services?\s+.*\bfailing\b",
            r"\bwhich\s+service\s+is\s+.*\bfail(?:ed|ing)\b",
            r"\bwhy\s+(are\s+there|is\s+there)\s+failed\s+services?\b",
        ],
    ),
    (
        ToolName.SERVICES_ALL,
        [
            r"\ball\s+services?\b",
            r"\ball\s+systemd\s+services?\b",
            r"\bavailable\s+services?\b",
            r"\bservices?\s+(are\s+)?available\b",
            r"\bwhat\s+systemd\s+services?\b",
            r"\bservices?\s+(exist|installed)\b",
            r"\bwhich\s+services?\s+(exist|are\s+installed)\b",
        ],
    ),
    (
        ToolName.LOGS_ERROR,
        [
            r"\brecent\s+errors?\b",
            r"\bany\s+errors?\b",
            r"\berror\s+logs?\b",
            r"\bshow\b.*\berrors?\b",
            r"\bwhat\s+errors?\b",
            r"\berror\s+messages?\b",
        ],
    ),
    (
        ToolName.LOGS_RECENT,
        [
            r"\brecent\s+logs?\b",
            r"\blatest\s+logs?\b",
            r"\bshow\b.*\blogs?\b",
            r"\bjournal\s*ctl\b",
            r"\bsystem\s+journal\b",
        ],
    ),
    (
        ToolName.DISK_USAGE,
        [
            r"\bwhich\s+partition\b",
            r"\bpartition(s)?\b.*\b(full|almost\s+full|space)\b",
            r"\bdisk\s+(space|usage)\b",
            r"\bdisk\s+almost\s+full\b",
            r"\bfree\s+disk\s+space\b",
            r"\bhow\s+full\b.*\bdisk\b",
        ],
    ),
    (
        ToolName.MEMORY_TOP,
        [
            r"\bmost\s+memory\b",
            r"\bhighest\s+memory\b",
            r"\bmemory\s+consuming\b",
            r"\bmemory\s+usage\b",
            r"\btop\b.*\bmemory\b",
            r"\bwhich\s+(process|application|app)\b.*\bmemory\b",
            r"\bconsuming\s+.*\bmemory\b",
            r"\bram\b",
            r"\bmemory\b",
        ],
    ),
    (
        ToolName.CPU_TOP,
        [
            r"\bmost\s+cpu\b",
            r"\bhighest\s+cpu\b",
            r"\bcpu\s+consuming\b",
            r"\bcpu\s+usage\b",
            r"\btop\b.*\bcpu\b",
            r"\bwhich\s+(process|application|app)\b.*\bcpu\b",
            r"\bconsuming\s+.*\bcpu\b",
            r"\bcpu\b",
        ],
    ),
    (
        ToolName.SERVICES_RUNNING,
        [
            r"\brunning\s+services?\b",
            r"\bshow\b.*\bservices?\b",
            r"\blist\b.*\bservices?\b",
            r"\bservices?\s+are\s+(currently\s+)?running\b",
            r"\bwhich\s+services?\s+are\s+(currently\s+)?running\b",
            r"\bwhich\s+services?\s+are\s+active\b",
            r"\bservices?\s+(are\s+)?active\b",
            r"\bhow\s+many\s+services?\b",
            r"\bcurrent\s+service\s+status\b",
            r"\bservice\s+status\b",
            r"\bgive\s+me\s+the\s+(current\s+)?service\s+status\b",
        ],
    ),
    (
        ToolName.NETWORK_PORTS,
        [
            r"\bopen\s+ports?\b",
            r"\blistening\s+ports?\b",
            r"\bnetwork\s+ports?\b",
            r"\bnetwork\s+connections?\b",
            r"\bwhich\s+ports?\b",
            r"\bsockets?\b",
        ],
    ),
]


def classify_ops_intent(message: str) -> OpsIntentResult | None:
    """Classify a message into a specific whitelisted tool, or None.

    Returns None (rather than a low-confidence guess) when nothing matches,
    so the caller knows to fall back to the general assistant pipeline
    instead of running a tool the question didn't actually ask for.
    """
    if not message or not message.strip():
        return None

    text = message.lower()

    # 1. Single-service questions take priority - "is nginx running?" is
    #    about nginx specifically, not the fleet-wide "running services" list.
    hint = _extract_service_hint(text)
    if hint is not None:
        svc, tool, needs_logs = hint
        confidence = 0.88 if needs_logs else 0.92
        return OpsIntentResult(
            tool=tool,
            matched_patterns=[f"service_hint:{svc}"],
            confidence=confidence,
            service_hint=svc,
            needs_logs=needs_logs,
        )

    # 2. Fleet-wide / generic ops patterns.
    for tool, patterns in _OPS_PATTERNS:
        matches = [p for p in patterns if re.search(p, text)]
        if matches:
            confidence = min(0.65 + 0.1 * len(matches), 0.97)
            return OpsIntentResult(tool=tool, matched_patterns=matches, confidence=confidence)

    return None
