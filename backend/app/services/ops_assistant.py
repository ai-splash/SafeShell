"""
Real-time Linux Operations Copilot - tool-grounded orchestration layer.

Pipeline for every ops query (see `ops_intent_classifier.classify_ops_intent`):
  1. Detect intent              -> already done by the caller (ops_intent_classifier)
  2. Execute a real, whitelisted, read-only command on the host -> tool_executor.execute_tool()
  3. Collect the structured result                                -> tool_executor.ToolResult
  4. Build a prompt grounded in that data                         -> ops_prompts.build_ops_prompt()
  5. Ask the LLM to answer using ONLY that data                   -> ollama_client.chat()
  6. Return summary + detailed structured results + confidence + suggested actions

This module never invents system data itself and never lets the LLM
override what the command actually returned - `detailed_results` in the
final response is always the tool's own structured output, independent of
whatever the model says in its summary.

Service-level intelligence (added): when `ops_intent_classifier` detects a
question about ONE specific service ("is nginx running?", "why did
postgres fail?"), the pipeline above is preceded by a resolution step
(`_resolve_service_unit`) that matches the free-text hint from the question
against the live list of systemd units on the host, and then gathers BOTH
that unit's structured status (`SERVICE_STATUS`) and its recent journal
entries (`SERVICE_LOGS`) in one round trip, so the LLM has everything it
needs to answer either a simple status question or a "why did it fail"
troubleshooting question from the same real data. See
`_process_service_query`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.config import get_settings
from app.logger import get_logger
from app.services import conversation_store
from app.services.llm_json import parse_json_object
from app.services.ollama_client import chat
from app.services.ops_intent_classifier import OpsIntentResult
from app.services.ops_prompts import (
    OPS_SYSTEM_PROMPT,
    build_ops_prompt,
    build_service_prompt,
)
from app.services.tool_executor import (
    ToolExecutionError,
    ToolName,
    execute_service_tool,
    execute_tool,
    normalize_unit_name,
)

settings = get_settings()
logger = get_logger(__name__)

# Tools handled by the composite single-service pipeline (resolve unit name
# against live data, then gather status + logs together) instead of the
# single fixed-tool pipeline below.
_SERVICE_TOOLS = (ToolName.SERVICE_STATUS, ToolName.SERVICE_LOGS)

# Best-effort aliases from common colloquial/product names to the systemd
# unit name(s) they're actually likely to correspond to. Used only to widen
# the search against the LIVE unit list in `_resolve_service_unit` - never
# used to fabricate a result if nothing on the host actually matches.
_SERVICE_ALIASES: dict[str, list[str]] = {
    "postgres": ["postgresql"],
    "postgress": ["postgresql"],
    "postgre": ["postgresql"],
    "psql": ["postgresql"],
    "mysql": ["mysql", "mysqld", "mariadb"],
    "maria": ["mariadb"],
    "mariadb": ["mariadb", "mysql"],
    "ssh": ["ssh", "sshd"],
    "sshd": ["sshd", "ssh"],
    "apache": ["apache2", "httpd"],
    "httpd": ["httpd", "apache2"],
    "redis": ["redis", "redis-server"],
    "cron": ["cron", "crond"],
    "web": ["nginx", "apache2", "httpd"],
    "webserver": ["nginx", "apache2", "httpd"],
    "docker": ["docker"],
    "nginx": ["nginx"],
}

# Safe, read-only fallback follow-up actions per tool, used only when the
# LLM itself returns no suggested_actions. Every entry is diagnostic/
# informational so it's always safe to surface even though the model didn't
# get a chance to tailor it.
_FALLBACK_ACTIONS: dict[str, list[dict]] = {
    ToolName.CPU_TOP.value: [
        {"command": "top -bn1 | head -20", "description": "Live snapshot of CPU usage across all processes.", "risk_level": "low"},
        {"command": "ps -p <pid> -o pid,ppid,cmd,%cpu,%mem", "description": "Inspect a specific process in more detail (replace <pid>).", "risk_level": "low"},
    ],
    ToolName.MEMORY_TOP.value: [
        {"command": "free -h", "description": "Overall RAM and swap usage.", "risk_level": "low"},
        {"command": "ps -p <pid> -o pid,ppid,cmd,%cpu,%mem", "description": "Inspect a specific process in more detail (replace <pid>).", "risk_level": "low"},
    ],
    ToolName.DISK_USAGE.value: [
        {"command": "du -ah / 2>/dev/null | sort -rh | head -20", "description": "Find the largest files/directories on the fullest filesystem.", "risk_level": "low"},
        {"command": "du -sh /var/log/* 2>/dev/null | sort -rh | head -10", "description": "Check whether logs are consuming the space.", "risk_level": "low"},
    ],
    ToolName.SERVICES_RUNNING.value: [
        {"command": "systemctl status <service-name>", "description": "Detailed status for a specific running service (replace <service-name>).", "risk_level": "low"},
    ],
    ToolName.SERVICES_FAILED.value: [
        {"command": "systemctl status <service-name>", "description": "See why a specific failed service failed (replace <service-name>).", "risk_level": "low"},
        {"command": "journalctl -u <service-name> --since '30 min ago'", "description": "Recent logs for a specific failed service (replace <service-name>).", "risk_level": "low"},
        {"command": "sudo systemctl restart <service-name>", "description": "Restart a specific failed service (replace <service-name>); briefly interrupts it.", "risk_level": "medium"},
    ],
    ToolName.LOGS_RECENT.value: [
        {"command": "journalctl -p err -n 50 --no-pager", "description": "Narrow the journal down to error-level entries only.", "risk_level": "low"},
    ],
    ToolName.LOGS_ERROR.value: [
        {"command": "journalctl -u <service-name> --since '1 hour ago'", "description": "Follow up on a specific service named in the errors (replace <service-name>).", "risk_level": "low"},
    ],
    ToolName.NETWORK_PORTS.value: [
        {"command": "ss -tulpn", "description": "Same listening sockets, plus which process owns each one.", "risk_level": "low"},
        {"command": "sudo ufw status verbose", "description": "Check whether the firewall is exposing these ports intentionally.", "risk_level": "low"},
    ],
    ToolName.SERVICES_ALL.value: [
        {"command": "systemctl --failed", "description": "Narrow the full service list down to just the ones that failed.", "risk_level": "low"},
        {"command": "systemctl status <service-name>", "description": "Detailed status for one specific service (replace <service-name>).", "risk_level": "low"},
    ],
    ToolName.SERVICE_STATUS.value: [
        {"command": "journalctl -u <service-name> --since '30 min ago'", "description": "Recent logs for this service (replace <service-name>).", "risk_level": "low"},
        {"command": "sudo systemctl restart <service-name>", "description": "Restart this service (replace <service-name>); briefly interrupts it.", "risk_level": "medium"},
    ],
    ToolName.SERVICE_LOGS.value: [
        {"command": "systemctl status <service-name>", "description": "Current structured status for this service (replace <service-name>).", "risk_level": "low"},
        {"command": "sudo systemctl restart <service-name>", "description": "Restart this service (replace <service-name>); briefly interrupts it.", "risk_level": "medium"},
    ],
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fallback_actions(tool_name: str) -> list[dict]:
    bank = _FALLBACK_ACTIONS.get(tool_name, [])
    return [dict(item) for item in bank]


def _build_history_messages(session_id: str) -> list[dict]:
    recent = conversation_store.get_recent_messages(
        session_id, limit=settings.assistant_history_turns * 2
    )
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def _resolve_service_unit(hint: str, known_units: list[dict]) -> tuple[str, bool]:
    """Resolve a free-text service hint (e.g. "nginx", "postgres", "the ssh
    service") to a real systemd unit name using the LIVE list of units
    collected from this host, where possible.

    Returns (unit_name, matched_in_unit_list). This is a NAME-RESOLUTION
    helper only - `matched_in_unit_list` tells the caller whether the
    chosen unit name happened to appear in the `SERVICES_ALL`
    (`systemctl list-units --type=service --all`) snapshot, which is
    useful for picking the right candidate among aliases and for
    diagnostics/logging. It is NOT the final word on whether the service
    is actually installed: `list-units --all` only lists units systemd has
    *loaded* at some point, so an installed-but-never-loaded unit won't
    appear there even though `systemctl show <unit>` would correctly
    report `LoadState=loaded` for it. Callers MUST determine actual
    existence from the `SERVICE_STATUS` (`systemctl show`) result's
    `parsed["found"]` field instead - see `_process_service_query`.
    """
    hint_clean = (hint or "").strip().lower().rstrip(".")
    for suffix in (".service", ".socket", ".target", ".timer"):
        if hint_clean.endswith(suffix):
            hint_clean = hint_clean[: -len(suffix)]
            break

    known_by_lower: dict[str, str] = {}
    for row in known_units:
        unit = row.get("unit") if isinstance(row, dict) else None
        if unit:
            known_by_lower[unit.lower()] = unit

    candidates = [hint_clean] + [c for c in _SERVICE_ALIASES.get(hint_clean, []) if c != hint_clean]

    # 1. Exact "<candidate>.service" match against the live unit list.
    for cand in candidates:
        exact = f"{cand}.service"
        if exact in known_by_lower:
            return known_by_lower[exact], True

    # 2. Live unit name starts with "<candidate>." or "<candidate>@" (e.g.
    #    a templated unit like "postgresql@14-main.service").
    for cand in candidates:
        if not cand:
            continue
        for lower_name, original in known_by_lower.items():
            if lower_name.startswith(f"{cand}.") or lower_name.startswith(f"{cand}@"):
                return original, True

    # 3. Candidate appears anywhere inside a live unit name.
    for cand in candidates:
        if not cand:
            continue
        for lower_name, original in known_by_lower.items():
            if cand in lower_name:
                return original, True

    # 4. No live match at all - fall back to the normalized guess, but flag
    #    it as unconfirmed so the prompt/response never claims it exists.
    normalized = normalize_unit_name(hint_clean) or f"{hint_clean}.service"
    return normalized, False


def _parse_ops_llm_response(raw: str, tool_name: str) -> tuple[dict, list[str]]:
    """Parse and validate the LLM's JSON reply for an ops/tool query.

    Mirrors `ai_assistant._parse_llm_response` but for the ops contract
    (`summary`/`suggested_actions` instead of `explanation`/
    `recommended_commands`). Never raises - a malformed reply degrades to a
    low-confidence summary instead of a 500.
    """
    warnings: list[str] = []

    try:
        data = parse_json_object(raw)
    except (ValueError,) as exc:
        warnings.append(f"Model response was not valid JSON ({exc}); showing raw text instead.")
        return (
            {
                "summary": raw.strip() or "The copilot did not return a usable response.",
                "suggested_actions": _fallback_actions(tool_name),
                "confidence_score": 0.2,
                "reasoning": "Response could not be parsed as structured JSON.",
            },
            warnings,
        )
    except Exception as exc:  # noqa: BLE001 - json.JSONDecodeError etc. from parse_json_object
        warnings.append(f"Model response was not valid JSON ({exc}); showing raw text instead.")
        return (
            {
                "summary": raw.strip() or "The copilot did not return a usable response.",
                "suggested_actions": _fallback_actions(tool_name),
                "confidence_score": 0.2,
                "reasoning": "Response could not be parsed as structured JSON.",
            },
            warnings,
        )

    summary = str(data.get("summary") or "").strip()
    if not summary:
        summary = "The copilot did not provide a summary."
        warnings.append("Missing 'summary' field in model response.")

    reasoning = str(data.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "No reasoning was provided by the model."
        warnings.append("Missing 'reasoning' field in model response.")

    try:
        confidence = float(data.get("confidence_score", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
        warnings.append("Invalid 'confidence_score'; defaulted to 0.5.")
    confidence = max(0.0, min(1.0, confidence))

    raw_actions = data.get("suggested_actions", [])
    actions: list[dict] = []
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict) or not item.get("command"):
                continue
            risk = str(item.get("risk_level", "low")).lower()
            if risk not in ("low", "medium", "high"):
                risk = "low"
            actions.append(
                {
                    "command": str(item["command"]),
                    "description": str(item.get("description", "")).strip() or "No description provided.",
                    "risk_level": risk,
                }
            )
    else:
        warnings.append("'suggested_actions' was not a list; ignored.")

    if not actions:
        actions = _fallback_actions(tool_name)
        if actions:
            warnings.append("Model returned no suggested actions; substituted safe defaults.")

    return (
        {
            "summary": summary,
            "suggested_actions": actions,
            "confidence_score": confidence,
            "reasoning": reasoning,
        },
        warnings,
    )


async def _process_service_query(message: str, session_id: str, ops_result: OpsIntentResult) -> dict:
    """Composite pipeline for a question about ONE specific service.

    1. Collect the live list of all known services (`SERVICES_ALL`) so the
       free-text hint from the question can be resolved against real unit
       names instead of guessed blindly.
    2. Resolve the hint to a unit name (`_resolve_service_unit`).
    3. Gather BOTH that unit's structured status and its recent journal
       entries - cheap, read-only, and gives the LLM everything it needs
       whether the question is a simple status check or a "why did it
       fail" troubleshooting question.
    4. Ground the LLM in that composite data and return the same response
       shape as the single-tool pipeline.
    """
    tool_name = ops_result.tool
    hint = ops_result.service_hint or ""

    logger.info(
        "Session %s: single-service ops intent -> tool=%s hint=%r needs_logs=%s",
        session_id,
        tool_name.value,
        hint,
        ops_result.needs_logs,
    )

    all_services_result = execute_tool(ToolName.SERVICES_ALL)
    known_units = all_services_result.parsed if isinstance(all_services_result.parsed, list) else []

    # `matched_in_unit_list` is only used to pick the best candidate unit
    # name (aliases, templated units, ...) - it is NOT the existence
    # determination. See `_resolve_service_unit` docstring.
    unit_name, matched_in_unit_list = _resolve_service_unit(hint, known_units)

    status_result = execute_service_tool(ToolName.SERVICE_STATUS, unit_name)
    logs_result = execute_service_tool(ToolName.SERVICE_LOGS, unit_name)

    # Authoritative existence check: `systemctl show <unit>` reports a real
    # LoadState for ANY validated unit name regardless of whether that unit
    # was ever loaded/started, so it is the source of truth - the live
    # `SERVICES_ALL` snapshot (`list-units --all`) only lists units that
    # happen to already be loaded and can miss an installed-but-never-loaded
    # service, which is exactly the false negative this fixes.
    status_parsed = status_result.parsed if isinstance(status_result.parsed, dict) else None
    if status_parsed is not None and status_parsed.get("load_state") not in (None, "", "unknown"):
        # `systemctl show` actually ran and returned a real LoadState
        # (loaded / not-found / masked / bad-setting / error / merged) -
        # trust it completely, in either direction, over the unit list.
        confirmed_installed = bool(status_parsed.get("found", False))
        existence_basis = "systemctl_show"
    else:
        # `systemctl show` didn't return usable data (command failed,
        # timed out, or systemd itself was unreachable) - we cannot make an
        # authoritative determination, so fall back to the live unit-list
        # match as a best-effort signal and say so explicitly rather than
        # asserting non-existence.
        confirmed_installed = matched_in_unit_list
        existence_basis = "live_unit_list_fallback"

    composite = {
        "requested_hint": hint,
        "resolved_unit": unit_name,
        "confirmed_installed": confirmed_installed,
        "existence_basis": existence_basis,
        "matched_in_unit_list": matched_in_unit_list,
        "status": status_result.to_dict(),
        "logs": logs_result.to_dict(),
    }

    history_messages = _build_history_messages(session_id)
    user_prompt = build_service_prompt(message, composite)
    llm_messages = (
        [{"role": "system", "content": OPS_SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_prompt}]
    )

    raw_reply = await chat(llm_messages)
    parsed, llm_warnings = _parse_ops_llm_response(raw_reply, tool_name.value)

    warnings = list(llm_warnings)
    if existence_basis == "systemctl_show" and not confirmed_installed:
        warnings.append(
            f"`systemctl show {unit_name}` reports LoadState="
            f"{(status_parsed or {}).get('load_state', 'unknown')!r} - "
            f"'{hint}' does not appear to be installed on this host."
        )
    elif existence_basis == "live_unit_list_fallback" and not confirmed_installed:
        warnings.append(
            f"Could not query live status for '{unit_name}' via `systemctl show` "
            f"(no usable LoadState returned), and '{hint}' was not found among the live "
            f"list of systemd services either; results below are a best-effort lookup "
            f"and the service may not exist."
        )
    elif existence_basis == "live_unit_list_fallback" and confirmed_installed:
        warnings.append(
            f"Could not query live status for '{unit_name}' via `systemctl show` "
            f"(no usable LoadState returned); existence is based only on the live "
            f"service list, not a direct status check."
        )
    if not all_services_result.success:
        warnings.append(f"Could not collect the live service list: {all_services_result.error}")
    if not status_result.success:
        warnings.append(f"Service status collection failed: {status_result.error}")
    if not logs_result.success:
        warnings.append(f"Service log collection failed: {logs_result.error}")
    warnings.extend(status_result.warnings)
    warnings.extend(logs_result.warnings)

    conversation_store.add_message(session_id, role="user", content=message)
    conversation_store.add_message(
        session_id,
        role="assistant",
        content=parsed["summary"],
        intent=tool_name.value,
        confidence_score=parsed["confidence_score"],
    )
    conversation_store.prune_session(session_id, keep_last=settings.assistant_max_history_stored)

    return {
        "session_id": session_id,
        "intent": tool_name.value,
        "explanation": parsed["summary"],
        "recommended_commands": parsed["suggested_actions"],
        "confidence_score": parsed["confidence_score"],
        "reasoning": parsed["reasoning"],
        "context_summary": {
            "tool": tool_name.value,
            "service_requested": hint,
            "service_resolved": unit_name,
            "confirmed_installed": confirmed_installed,
            "existence_basis": existence_basis,
            "status_command": status_result.display_command,
            "logs_command": logs_result.display_command,
            "success": status_result.success and logs_result.success,
        },
        "warnings": warnings,
        "timestamp": _now_iso(),
        "tool_used": tool_name.value,
        "detailed_results": composite,
    }


async def process_ops_query(message: str, session_id: str | None, ops_result: OpsIntentResult) -> dict:
    """Run the full tool-grounded ops pipeline for a single user query.

    Returns a dict matching the extended ChatResponse schema (adds
    `tool_used` and `detailed_results` on top of the original fields).
    Raises `OllamaUnavailableError` if the LLM cannot be reached - callers
    (routes) should translate that into an HTTP 503, same as the original
    assistant pipeline.
    """
    session_id = session_id or str(uuid.uuid4())
    tool_name = ops_result.tool

    # Single-service questions ("is nginx running?", "why did postgres
    # fail?") use the composite resolve+status+logs pipeline instead of the
    # single fixed-tool pipeline below.
    if tool_name in _SERVICE_TOOLS and ops_result.service_hint:
        return await _process_service_query(message, session_id, ops_result)

    logger.info(
        "Session %s: ops intent -> tool=%s (matched=%s, confidence=%.2f)",
        session_id,
        tool_name.value,
        ops_result.matched_patterns,
        ops_result.confidence,
    )

    # 2 & 3. Execute the whitelisted tool and collect its structured result.
    try:
        tool_result = execute_tool(tool_name)
    except ToolExecutionError as exc:
        # Only raised for an unrecognized tool name - a programmer error in
        # the classifier/registry, not something a live host can trigger.
        logger.error("Tool execution rejected: %s", exc)
        raise

    tool_dict = tool_result.to_dict()

    # 4. Build the ops prompt grounded in that real data.
    history_messages = _build_history_messages(session_id)
    user_prompt = build_ops_prompt(message, tool_name.value, tool_result.display_command, tool_dict)
    llm_messages = (
        [{"role": "system", "content": OPS_SYSTEM_PROMPT}]
        + history_messages
        + [{"role": "user", "content": user_prompt}]
    )

    # 5. Ask the LLM to answer using only that data (raises OllamaUnavailableError on failure).
    raw_reply = await chat(llm_messages)

    # 6. Parse into summary / suggested actions / confidence / reasoning.
    parsed, llm_warnings = _parse_ops_llm_response(raw_reply, tool_name.value)

    warnings = list(llm_warnings)
    if not tool_result.success:
        warnings.append(f"Tool execution failed: {tool_result.error}")
    warnings.extend(tool_result.warnings)

    # Persist conversation turns exactly like the original assistant pipeline.
    conversation_store.add_message(session_id, role="user", content=message)
    conversation_store.add_message(
        session_id,
        role="assistant",
        content=parsed["summary"],
        intent=tool_name.value,
        confidence_score=parsed["confidence_score"],
    )
    conversation_store.prune_session(session_id, keep_last=settings.assistant_max_history_stored)

    return {
        "session_id": session_id,
        "intent": tool_name.value,
        "explanation": parsed["summary"],
        "recommended_commands": parsed["suggested_actions"],
        "confidence_score": parsed["confidence_score"],
        "reasoning": parsed["reasoning"],
        "context_summary": {
            "tool": tool_name.value,
            "command": tool_result.display_command,
            "row_count": tool_result.row_count,
            "success": tool_result.success,
        },
        "warnings": warnings,
        "timestamp": _now_iso(),
        "tool_used": tool_name.value,
        "detailed_results": tool_dict,
    }
