"""
Intent classification for the AI Ops Assistant.

A lightweight, fast, fully offline keyword/pattern classifier. It decides
*what kind* of system context needs to be gathered before we ever talk to
the LLM - e.g. "why is my system slow?" needs CPU/memory/process data,
"restart nginx" needs service status.

Deliberately rule-based (not a second LLM call): it's instant, free, and
"good enough" for routing context. The LLM still does all the actual
reasoning and natural-language understanding for the final answer - this
step only decides which system facts are worth fetching first.
"""

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    PERFORMANCE = "performance"              # "why is my system slow?"
    SERVICE_MANAGEMENT = "service_management"  # "restart nginx", "is ssh running?"
    DOCKER = "docker"                          # "show running docker containers"
    FILE_SEARCH = "file_search"                # "find large files"
    LOG_ANALYSIS = "log_analysis"               # "explain this error", "check logs"
    NETWORK = "network"                         # "why is my network slow", "check open ports"
    USERS_SESSIONS = "users_sessions"           # "who is logged in"
    GENERAL = "general"                         # fallback: general Linux Q&A


@dataclass
class IntentResult:
    intent: Intent
    matched_keywords: list[str]
    confidence: float  # heuristic confidence in the classification itself


# Ordered so more specific intents are checked before generic ones.
_INTENT_PATTERNS: list[tuple[Intent, list[str]]] = [
    (
        Intent.DOCKER,
        [
            r"\bdocker\b",
            r"\bcontainer(s)?\b",
            r"\bdocker[- ]compose\b",
            r"\bimage(s)?\b.*\bdocker\b",
        ],
    ),
    (
        Intent.SERVICE_MANAGEMENT,
        [
            r"\brestart\b",
            r"\bstop\b.*\bservice\b",
            r"\bstart\b.*\bservice\b",
            r"\bsystemctl\b",
            r"\bnginx\b",
            r"\bapache\b",
            r"\bservice\s+status\b",
            r"\bis\s+\w+\s+running\b",
            r"\benable\b.*\bservice\b",
        ],
    ),
    (
        Intent.FILE_SEARCH,
        [
            r"\blarge\s+files?\b",
            r"\bdisk\s+space\b",
            r"\bwhat.?s\s+using\s+(my\s+)?disk\b",
            r"\bfind\s+files?\b",
            r"\bfree\s+up\s+space\b",
        ],
    ),
    (
        Intent.LOG_ANALYSIS,
        [
            r"\berror\b",
            r"\bexception\b",
            r"\bstack\s?trace\b",
            r"\blogs?\b",
            r"\bjournal\b",
            r"\bwhy\s+did\s+.*\bfail\b",
            r"\bcrash(ed)?\b",
            r"\bexplain\s+this\b",
        ],
    ),
    (
        Intent.NETWORK,
        [
            r"\bnetwork\b",
            r"\bbandwidth\b",
            r"\bport(s)?\b",
            r"\bconnection(s)?\b",
            r"\bping\b",
            r"\bfirewall\b",
            r"\binternet\b",
        ],
    ),
    (
        Intent.USERS_SESSIONS,
        [
            r"\bwho\s+is\s+logged\s+in\b",
            r"\blogged.?in\s+users?\b",
            r"\bactive\s+sessions?\b",
            r"\bwho\s+is\s+on\s+(this|the)\s+system\b",
        ],
    ),
    (
        Intent.PERFORMANCE,
        [
            r"\bslow\b",
            r"\bhigh\s+cpu\b",
            r"\bcpu\s+usage\b",
            r"\bmemory\s+usage\b",
            r"\bram\b",
            r"\bload\s+average\b",
            r"\bperformance\b",
            r"\blagg(y|ing)\b",
            r"\bfreez(e|ing)\b",
            r"\bhang(ing|s)?\b",
        ],
    ),
]


def classify_intent(message: str) -> IntentResult:
    """Classify a natural-language message into an operational intent.

    Runs each intent's patterns against the message (case-insensitive) and
    returns the first intent with at least one match, ordered by
    specificity. Falls back to GENERAL if nothing matches.
    """
    if not message or not message.strip():
        return IntentResult(intent=Intent.GENERAL, matched_keywords=[], confidence=0.0)

    text = message.lower()

    for intent, patterns in _INTENT_PATTERNS:
        matches = [p for p in patterns if re.search(p, text)]
        if matches:
            # Confidence scales gently with number of distinct pattern hits,
            # capped so it never claims false certainty.
            confidence = min(0.6 + 0.1 * len(matches), 0.95)
            return IntentResult(intent=intent, matched_keywords=matches, confidence=confidence)

    return IntentResult(intent=Intent.GENERAL, matched_keywords=[], confidence=0.4)
