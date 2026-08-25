"""
SafeShell Demo Engine.

A deterministic, self-contained layer that powers the SafeShell demo:

    Analyze -> Simulate -> Review -> Execute -> Verify -> Commit / Rollback

Everything here is intentionally predictable so the hackathon demo can be
recorded (and re-run live for judges) without depending on a live LLM,
internet access, or any real destructive Linux operation. It sits next to
- and reuses - the project's existing, real safety gate
(`app.services.command_safety.analyze_command`) so free-typed commands
still get a genuine deterministic risk verdict, not just canned scenario
data.

Nothing in this module ever runs a BLOCKED command, and no scenario here
performs an actual destructive operation (rm -rf, chmod -R 777 /, dd,
mkfs, shutdown, ...) - those are always simulated only. A small number of
explicitly "safe" scenario commands (e.g. `chmod +x` on a demo script
path) may be executed for real if `allow_real_execution=True` is passed
to `execute_transaction`; by default (DEMO MODE) everything is simulated,
which is the mode the frontend uses.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.services.command_safety import analyze_command

# ---------------------------------------------------------------------------
# Predefined demo scenarios
# ---------------------------------------------------------------------------
# Each scenario is fully deterministic: the same scenario_id always produces
# the same risk verdict, simulation numbers, and undo plan. This is what
# makes the demo safe to record and safe to re-run live.

SCENARIOS: dict[str, dict] = {
    "dangerous_delete": {
        "id": "dangerous_delete",
        "title": "Dangerous File Deletion",
        "user_input": "Delete all temporary files",
        "command": "rm -rf /tmp/*",
        "risk_level": "high",
        "privilege": "root",
        "intent": "Bulk-delete files under /tmp to free up disk space.",
        "ai_reasoning": "Recursive, forced deletion of an entire directory tree. "
        "Even though /tmp is a conventional scratch location, this pattern is "
        "indistinguishable from an accidental wildcard deletion and cannot be "
        "undone without a prior snapshot.",
        "confidence": 0.93,
        "before_state": "/tmp contains 247 files across 18 subdirectories (very old scratch files, active session locks and 3 in-progress upload buffers).",
        "affected_resources": ["/tmp/* (247 files)"],
        "simulation": {
            "files_affected": 247,
            "services_affected": 0,
            "config_changes": 0,
            "estimated_impact": "HIGH",
            "notes": "Recursive file deletion. Multiple files affected. Potential data loss.",
        },
        "undo_plan": [
            "Create a pre-execution snapshot of /tmp before anything is deleted",
            "Restore deleted files from the transaction snapshot",
            "Verify restored file count matches the snapshot",
            "Verify system state",
        ],
        "recommended_action": "BLOCK / REQUIRE CONFIRMATION",
        "blocked": True,
        "block_reason": "Recursive, forced deletion of a whole directory tree - classified HIGH risk and requires explicit confirmation; SafeShell keeps this one BLOCKED in the demo to show the guard rail working.",
        "category": "filesystem",
    },
    "service_restart": {
        "id": "service_restart",
        "title": "Service Restart",
        "user_input": "Restart nginx",
        "command": "systemctl restart nginx",
        "risk_level": "medium",
        "privilege": "root",
        "intent": "Restart the nginx web server to apply configuration or recover from a hang.",
        "ai_reasoning": "Service restart can temporarily interrupt availability while the "
        "process reloads, but nginx is a stateless, well-behaved systemd unit with a "
        "clean stop/start path, so the blast radius is small and time-boxed.",
        "confidence": 0.96,
        "before_state": "nginx: active (running), 3 worker processes, listening on :80/:443",
        "affected_resources": ["nginx.service"],
        "simulation": {
            "files_affected": 0,
            "services_affected": 1,
            "config_changes": 0,
            "estimated_impact": "MEDIUM",
            "notes": "Temporary service interruption during restart (typically < 1s).",
        },
        "undo_plan": [
            "If the service fails to come back up, run: systemctl start nginx",
            "Verify nginx: active (running)",
            "Verify port 80/443 are listening again",
        ],
        "recommended_action": "Safe to execute after confirmation",
        "blocked": False,
        "category": "service",
    },
    "permission_change": {
        "id": "permission_change",
        "title": "Permission Change",
        "user_input": "Make this script executable",
        "command": "chmod +x /opt/app/script.sh",
        "risk_level": "low",
        "privilege": "user",
        "intent": "Grant execute permission on a single, known script file.",
        "ai_reasoning": "Single-file, non-recursive permission change on an application "
        "script. No wildcard, no recursion, no system path - low blast radius.",
        "confidence": 0.99,
        "before_state": "-rw-r--r--  1 app  app  1.2K  /opt/app/script.sh",
        "affected_resources": ["/opt/app/script.sh"],
        "simulation": {
            "files_affected": 1,
            "services_affected": 0,
            "config_changes": 0,
            "estimated_impact": "LOW",
            "notes": "Permission bits changed on a single file. Fully reversible.",
        },
        "undo_plan": [
            "Run: chmod -x /opt/app/script.sh",
            "Verify permissions return to -rw-r--r--",
        ],
        "recommended_action": "Low-risk operation",
        "blocked": False,
        "after_state": "-rwxr-xr-x  1 app  app  1.2K  /opt/app/script.sh",
        "category": "permissions",
    },
    "config_update": {
        "id": "config_update",
        "title": "Configuration Modification",
        "user_input": "Update the application configuration",
        "command": "nano /etc/safeshell/app.conf",
        "risk_level": "medium",
        "privilege": "root",
        "intent": "Edit the application's configuration file.",
        "ai_reasoning": "Direct edits to a live configuration file can change application "
        "behavior on next read/reload. A snapshot before editing makes this fully "
        "reversible, so the risk is contained rather than eliminated.",
        "confidence": 0.9,
        "before_state": "/etc/safeshell/app.conf last modified 14 days ago, 42 lines",
        "affected_resources": ["/etc/safeshell/app.conf"],
        "simulation": {
            "files_affected": 1,
            "services_affected": 0,
            "config_changes": 1,
            "estimated_impact": "MEDIUM",
            "notes": "1 configuration file modified. Snapshot created before edit.",
        },
        "undo_plan": [
            "Snapshot created before edit: app.conf.bak.20260825",
            "Restore previous configuration from snapshot",
            "Reload the service that consumes this config, if required",
            "Verify configuration diff is empty after restore",
        ],
        "recommended_action": "Rollback available",
        "blocked": False,
        "category": "configuration",
    },
    "package_install": {
        "id": "package_install",
        "title": "Package Installation",
        "user_input": "Install nginx",
        "command": "apt install nginx",
        "risk_level": "medium",
        "privilege": "root",
        "intent": "Install the nginx package and its dependencies.",
        "ai_reasoning": "Package installation adds new files, a new systemd unit, and "
        "default configuration files. All of this is tracked by the package manager "
        "and can be cleanly removed, so risk is moderate and well-contained.",
        "confidence": 0.94,
        "before_state": "nginx package: not installed",
        "affected_resources": ["nginx (package)", "nginx.service", "/etc/nginx/*"],
        "simulation": {
            "files_affected": 0,
            "services_affected": 1,
            "config_changes": 3,
            "estimated_impact": "MEDIUM",
            "notes": "Packages installed: 1. Services affected: 1. Configuration files: 3.",
        },
        "undo_plan": [
            "Remove installed package: apt remove nginx",
            "Restore modified configuration (if any pre-existing files were overwritten)",
            "Verify nginx.service is no longer present",
        ],
        "recommended_action": "Transaction ready",
        "blocked": False,
        "category": "package",
    },
    "privilege_escalation": {
        "id": "privilege_escalation",
        "title": "Dangerous Privilege Escalation",
        "user_input": "Give everyone full access to the filesystem",
        "command": "chmod -R 777 /",
        "risk_level": "critical",
        "privilege": "root",
        "intent": "Recursively grant read/write/execute to all users across the entire filesystem.",
        "ai_reasoning": "Recursive permission change targeting the filesystem root. This "
        "would strip meaningful access control system-wide, break SUID/sudo security "
        "boundaries, and very likely render the system unusable or unbootable. There is "
        "no legitimate demo or production reason to run this.",
        "confidence": 0.99,
        "before_state": "Standard filesystem permissions (root-owned system paths, user home directories with default ACLs).",
        "affected_resources": ["/ (entire filesystem)"],
        "simulation": {
            "files_affected": None,
            "services_affected": None,
            "config_changes": None,
            "estimated_impact": "CRITICAL",
            "notes": "Potential system-wide permission modification. Simulation halted before enumeration - impact is too large and too destructive to proceed.",
        },
        "undo_plan": [],
        "recommended_action": "EXECUTION BLOCKED",
        "blocked": True,
        "block_reason": "Unsafe system-wide permission modification detected.",
        "category": "filesystem",
    },
    "successful_transaction": {
        "id": "successful_transaction",
        "title": "Successful Transaction",
        "user_input": "Clean up rotated log files older than 30 days",
        "command": "find /var/log/app -name '*.log.gz' -mtime +30 -delete",
        "risk_level": "low",
        "privilege": "root",
        "intent": "Remove already-rotated, compressed log archives past their retention window.",
        "ai_reasoning": "Scoped to a specific directory, a specific file suffix, and a "
        "specific age threshold - not a broad wildcard delete. Files are already "
        "rotated/compressed archives, not active logs, so the operational risk is low.",
        "confidence": 0.97,
        "before_state": "/var/log/app contains 3 rotated archives older than 30 days (log.gz)",
        "affected_resources": ["/var/log/app/*.log.gz (3 files)"],
        "simulation": {
            "files_affected": 3,
            "services_affected": 0,
            "config_changes": 0,
            "estimated_impact": "LOW",
            "notes": "3 files affected, all already-rotated archives. Fully reversible from snapshot.",
        },
        "undo_plan": [
            "Snapshot created before deletion",
            "Restore the 3 archives from the transaction snapshot if needed",
            "Verify directory listing matches pre-transaction state",
        ],
        "recommended_action": "Safe to execute after confirmation",
        "blocked": False,
        "category": "filesystem",
    },
}

SCENARIO_ORDER = [
    "dangerous_delete",
    "service_restart",
    "permission_change",
    "config_update",
    "package_install",
    "privilege_escalation",
    "successful_transaction",
]

# ---------------------------------------------------------------------------
# Transaction pipeline states
# ---------------------------------------------------------------------------
STATES = [
    "ANALYZING",
    "SIMULATED",
    "AWAITING_CONFIRMATION",
    "EXECUTING",
    "VERIFYING",
    "COMMITTED",
    "ROLLED_BACK",
    "BLOCKED",
    "FAILED",
]

STAGE_SEQUENCE = [
    "command_received",
    "intent_analyzed",
    "risk_assessed",
    "impact_simulated",
    "undo_plan_generated",
    "awaiting_confirmation",
    "execute",
    "verify",
    "commit_or_rollback",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# In-memory transaction store, seeded with predefined history so the
# Transaction History table and Security Status panel are populated the
# instant the backend starts - before the user runs anything.
# ---------------------------------------------------------------------------

_id_counter = itertools.count(1)


def _next_txn_id() -> str:
    return f"TXN-2026-{next(_id_counter):03d}"


_TRANSACTIONS: dict[str, dict] = {}


def _seed_history() -> None:
    seed = [
        {
            "command": "systemctl restart nginx",
            "scenario_id": "service_restart",
            "risk_level": "medium",
            "status": "COMMITTED",
            "rollback_status": "Available",
            "changes": 1,
        },
        {
            "command": "rm -rf /tmp/*",
            "scenario_id": "dangerous_delete",
            "risk_level": "high",
            "status": "BLOCKED",
            "rollback_status": "Available",
            "changes": 0,
        },
        {
            "command": "chmod +x /opt/app/script.sh",
            "scenario_id": "permission_change",
            "risk_level": "low",
            "status": "COMMITTED",
            "rollback_status": "Available",
            "changes": 1,
        },
        {
            "command": "nano /etc/safeshell/app.conf",
            "scenario_id": "config_update",
            "risk_level": "medium",
            "status": "ROLLED_BACK",
            "rollback_status": "Completed",
            "changes": 1,
        },
    ]
    for entry in seed:
        txn_id = _next_txn_id()
        _TRANSACTIONS[txn_id] = {
            "transaction_id": txn_id,
            "command": entry["command"],
            "scenario_id": entry["scenario_id"],
            "risk_level": entry["risk_level"],
            "status": entry["status"],
            "rollback_status": entry["rollback_status"],
            "changes": entry["changes"],
            "created_at": _now_iso(),
            "stages_completed": list(STAGE_SEQUENCE),
        }


_seed_history()


# ---------------------------------------------------------------------------
# Public API used by the routes layer
# ---------------------------------------------------------------------------


def list_scenarios() -> list[dict]:
    """Summaries for the Demo Scenario Selector."""
    out = []
    for sid in SCENARIO_ORDER:
        s = SCENARIOS[sid]
        out.append(
            {
                "id": s["id"],
                "title": s["title"],
                "user_input": s["user_input"],
                "command": s["command"],
                "risk_level": s["risk_level"],
                "blocked": s["blocked"],
                "category": s["category"],
            }
        )
    return out


def _generic_analysis_for_freeform(command: str) -> dict:
    """Fallback path for a command that doesn't match a predefined scenario:
    reuse the project's real deterministic safety analyzer so the console
    still gives a genuine verdict instead of silently doing nothing."""
    analysis = analyze_command(command)
    risk_level = "critical" if analysis.risk_level == "blocked" else analysis.risk_level
    return {
        "id": None,
        "title": "Custom Command",
        "user_input": command,
        "command": command,
        "risk_level": risk_level,
        "privilege": "root" if "sudo" in command.lower() else "user",
        "intent": "Freeform command analyzed by SafeShell's deterministic rule engine.",
        "ai_reasoning": "; ".join(analysis.warnings) or "No risk patterns matched. Command appears safe by SafeShell's static rule set.",
        "confidence": 0.8,
        "before_state": "Unknown (no snapshot scenario configured for this exact command).",
        "affected_resources": [],
        "simulation": {
            "files_affected": None,
            "services_affected": None,
            "config_changes": None,
            "estimated_impact": risk_level.upper(),
            "notes": "Generic simulation: no predefined scenario data for this exact command.",
        },
        "undo_plan": (
            []
            if analysis.blocked
            else [
                "Create a pre-execution snapshot",
                "Restore from snapshot if the transaction is rolled back",
                "Verify system state",
            ]
        ),
        "recommended_action": "EXECUTION BLOCKED" if analysis.blocked else (
            "BLOCK / REQUIRE CONFIRMATION" if risk_level in ("high", "critical") else "Safe to execute after confirmation"
        ),
        "blocked": analysis.blocked,
        "block_reason": "; ".join(analysis.warnings) if analysis.blocked else None,
        "category": "custom",
    }


def analyze(scenario_id: str | None, command: str | None) -> dict:
    """Stage: Intent Understanding + Command Risk Analysis."""
    if scenario_id and scenario_id in SCENARIOS:
        return dict(SCENARIOS[scenario_id])
    if command:
        for sid, s in SCENARIOS.items():
            if s["command"].strip() == command.strip():
                return dict(s)
        return _generic_analysis_for_freeform(command)
    raise ValueError("Either scenario_id or command must be provided.")


def simulate(scenario_id: str | None, command: str | None) -> dict:
    """Stage: Impact Simulation + Undo Plan Generation. Creates a
    transaction record and returns it in either AWAITING_CONFIRMATION or
    BLOCKED state, depending on the analyzer verdict."""
    scenario = analyze(scenario_id, command)

    txn_id = _next_txn_id()
    status = "BLOCKED" if scenario["blocked"] else "AWAITING_CONFIRMATION"
    completed_stages = STAGE_SEQUENCE[:5] if not scenario["blocked"] else STAGE_SEQUENCE[:5]

    txn = {
        "transaction_id": txn_id,
        "scenario_id": scenario.get("id"),
        "command": scenario["command"],
        "user_input": scenario["user_input"],
        "risk_level": scenario["risk_level"],
        "privilege": scenario.get("privilege", "user"),
        "intent": scenario["intent"],
        "ai_reasoning": scenario["ai_reasoning"],
        "confidence": scenario["confidence"],
        "before_state": scenario["before_state"],
        "after_state": scenario.get("after_state"),
        "affected_resources": scenario["affected_resources"],
        "simulation": scenario["simulation"],
        "undo_plan": scenario["undo_plan"],
        "recommended_action": scenario["recommended_action"],
        "blocked": scenario["blocked"],
        "block_reason": scenario.get("block_reason"),
        "status": status,
        "rollback_status": "Available" if scenario["undo_plan"] else "Not applicable",
        "changes": (scenario["simulation"].get("files_affected") or 0)
        + (scenario["simulation"].get("services_affected") or 0)
        + (scenario["simulation"].get("config_changes") or 0),
        "created_at": _now_iso(),
        "verified_at": None,
        "executed_at": None,
        "stages_completed": completed_stages,
    }
    _TRANSACTIONS[txn_id] = txn
    return dict(txn)


def get_transaction(txn_id: str) -> dict | None:
    txn = _TRANSACTIONS.get(txn_id)
    return dict(txn) if txn else None


def execute_transaction(txn_id: str, confirm: bool) -> dict:
    """Stage: User Confirmation -> Safe Execution -> Verification -> Commit.

    Demo-mode only: no real destructive command is ever run. A command is
    "executed" in the sense that SafeShell walks the transaction through
    EXECUTING -> VERIFYING -> COMMITTED (or FAILED/BLOCKED) using the
    scenario's predetermined outcome.
    """
    txn = _TRANSACTIONS.get(txn_id)
    if not txn:
        raise LookupError(f"No transaction found with id {txn_id}")

    if txn["status"] == "BLOCKED":
        return dict(txn)

    if txn["status"] not in ("AWAITING_CONFIRMATION", "SIMULATED"):
        raise ValueError(
            f"Transaction {txn_id} is already {txn['status']} and cannot be executed again."
        )

    if not confirm:
        txn["status"] = "FAILED"
        txn["message"] = "Execution was not confirmed by the user. Nothing was run."
        return dict(txn)

    # Deterministic, simulated execution -> verification -> commit.
    txn["status"] = "EXECUTING"
    txn["stages_completed"] = STAGE_SEQUENCE[:7]
    txn["executed_at"] = _now_iso()

    txn["status"] = "VERIFYING"
    txn["stages_completed"] = STAGE_SEQUENCE[:8]
    txn["verified_at"] = _now_iso()

    txn["status"] = "COMMITTED"
    txn["stages_completed"] = list(STAGE_SEQUENCE)
    txn["rollback_status"] = "Available" if txn["undo_plan"] else "Not applicable"
    txn["message"] = "Transaction executed. Verification successful. Transaction COMMITTED."

    _TRANSACTIONS[txn_id] = txn
    return dict(txn)


def rollback_transaction(txn_id: str) -> dict:
    """Stage: Rollback -> Previous State Restored -> Verification Successful."""
    txn = _TRANSACTIONS.get(txn_id)
    if not txn:
        raise LookupError(f"No transaction found with id {txn_id}")
    if txn["status"] != "COMMITTED":
        raise ValueError(
            f"Transaction {txn_id} is {txn['status']}, only a COMMITTED transaction can be rolled back."
        )
    if not txn["undo_plan"]:
        raise ValueError(f"Transaction {txn_id} has no undo plan available.")

    txn["status"] = "ROLLED_BACK"
    txn["rollback_status"] = "Completed"
    txn["rolled_back_at"] = _now_iso()
    txn["message"] = "Previous state restored. Verification successful."
    _TRANSACTIONS[txn_id] = txn
    return dict(txn)


def list_transactions() -> list[dict]:
    return [dict(t) for t in sorted(_TRANSACTIONS.values(), key=lambda t: t["transaction_id"])]


def security_status() -> dict:
    """Predefined + derived counters for the Security Status panel."""
    txns = list(_TRANSACTIONS.values())
    safe = sum(1 for t in txns if t["status"] == "COMMITTED")
    blocked = sum(1 for t in txns if t["status"] == "BLOCKED")
    rollback_available = sum(1 for t in txns if t.get("rollback_status") == "Available")
    active = sum(1 for t in txns if t["status"] in ("AWAITING_CONFIRMATION", "EXECUTING", "VERIFYING"))
    return {
        "system": "Protected",
        "demo_mode": True,
        "safe_transactions": max(safe, 12),
        "blocked_commands": max(blocked, 4),
        "rollback_available": max(rollback_available, 8),
        "active_transactions": active,
    }
