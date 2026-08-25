"""
Prompt templates for the real-time Linux Operations Copilot tool pipeline.

Distinct from `prompts.py` (the original "explain + recommend commands,
no data" assistant): every prompt built here is grounded in the output of
a command that was *just executed* on the live host by `tool_executor.py`.
The system prompt below tells the model exactly that, and forbids it from
answering with anything not present in that data.
"""

from __future__ import annotations

import json

# Sent on every ops-tool request. Kept short for the same latency reasons as
# prompts.SYSTEM_PROMPT, but the contract is different: "summary" instead of
# "explanation", and the rules make explicit that the data is live and real.
OPS_SYSTEM_PROMPT = """You are Linux Copilot XAI, a real-time Linux Operations Copilot.

A read-only Linux command was just executed on the live host and its parsed output is given to \
you below as CURRENT SYSTEM DATA. Rules: answer ONLY using that data - never invent processes, \
services, service states, PIDs, numbers, or log lines that are not present in it. If the data is \
empty or an error is noted, say so plainly and lower confidence_score. Reference real names/numbers \
from the data in your summary. Suggested follow-up commands must be safe; if a command would change \
system state (restart/stop/kill/delete/edit/enable/disable/install), mark it risk_level "medium" or \
"high" and say what it would do.

For single-service questions, the data may include a 'confirmed_installed_on_host' flag: if it is \
false, tell the user you could not confirm that service exists on this host rather than asserting it \
is installed. When both a service's current status and its recent logs are provided, clearly separate \
confirmed facts (the reported state) from your inference about a likely cause, and don't claim a root \
cause the logs don't actually support.

Respond with ONLY this JSON object, no markdown fences, no other text:
{"summary": "<2-3 plain-language sentences answering the user question, citing real data>", \
"suggested_actions": [{"command": "<shell command>", "description": "<why>", \
"risk_level": "low|medium|high"}], "confidence_score": <0.0-1.0>, \
"reasoning": "<brief basis for the answer, referencing the data>"}

Use an empty list for suggested_actions if none apply. All four fields are required."""


def build_tool_data_block(tool_name: str, display_command: str, tool_result: dict) -> str:
    """Render one tool's structured result into a compact text block."""
    payload = {
        "tool": tool_name,
        "command_executed": display_command,
        "success": tool_result.get("success"),
        "row_count": tool_result.get("row_count"),
        "data": tool_result.get("parsed"),
        "error": tool_result.get("error"),
        "warnings": tool_result.get("warnings") or [],
    }
    try:
        rendered = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(payload)
    return rendered


def build_ops_prompt(message: str, tool_name: str, display_command: str, tool_result: dict) -> str:
    """Compose the final user-turn prompt for a single-tool, fleet-wide
    ops/tool-grounded query (top CPU/memory, disk, running/failed services,
    logs, ports, all-services).

    Follows the required shape: current system data, then the user
    question, then an explicit instruction to answer only from that data.
    """
    data_block = build_tool_data_block(tool_name, display_command, tool_result)
    return (
        f"Current System Data:\n{data_block}\n\n"
        f"User Question: {message}\n\n"
        "Answer only using the provided system data above. "
        "Respond with the JSON object described in your instructions."
    )


def build_service_data_block(composite: dict) -> str:
    """Render the composite single-service result (status + recent logs,
    plus how the service name was resolved) into a compact text block.

    `composite` is built by `ops_assistant._process_service_query` and has
    the shape: {requested_hint, resolved_unit, confirmed_installed,
    status: ToolResult.to_dict(), logs: ToolResult.to_dict()}.
    """
    status = composite.get("status") or {}
    logs = composite.get("logs") or {}
    payload = {
        "requested_hint": composite.get("requested_hint"),
        "resolved_unit": composite.get("resolved_unit"),
        "confirmed_installed_on_host": composite.get("confirmed_installed"),
        "status": {
            "command_executed": status.get("command"),
            "success": status.get("success"),
            "data": status.get("parsed"),
            "error": status.get("error"),
            "warnings": status.get("warnings") or [],
        },
        "recent_logs": {
            "command_executed": logs.get("command"),
            "success": logs.get("success"),
            "lines": logs.get("parsed"),
            "error": logs.get("error"),
            "warnings": logs.get("warnings") or [],
        },
    }
    try:
        rendered = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        rendered = str(payload)
    return rendered


def build_service_prompt(message: str, composite: dict) -> str:
    """Compose the final user-turn prompt for a single-service question
    ("is nginx running?", "why did postgres fail?", ...).

    Grounds the model in BOTH the service's current structured status
    (`systemctl show`) and its recent journal entries in one shot, since
    troubleshooting questions need both to separate evidence from
    inference, while pure status questions can just ignore the log lines.
    """
    data_block = build_service_data_block(composite)
    return (
        f"Current Service Data:\n{data_block}\n\n"
        f"User Question: {message}\n\n"
        "'status' is the live, structured systemd state for the resolved unit (load_state/"
        "active_state/sub_state/enabled/main_pid/result/active_since). 'recent_logs' is its most "
        "recent journal entries, useful for explaining WHY it is in that state. If "
        "'confirmed_installed_on_host' is false, the service could not be found among the live "
        "list of installed services - say so plainly instead of asserting it exists. "
        "Answer only using the provided data above. "
        "Respond with the JSON object described in your instructions."
    )
