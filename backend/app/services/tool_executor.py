"""
Tool Executor - the "hands" of the Linux Operations Copilot.

This is the ONLY module that turns an ops question into a real command run
on the host. It is deliberately narrow and paranoid:

  * Every runnable command is a fixed, hard-coded argv list defined in
    ``_TOOLS`` below. There is no code path that builds a command from user
    input, string-formats a shell string, or accepts a caller-supplied
    command. Callers can only ever ask for one of the named tools
    (``ToolName``) - there is nothing to inject.
  * A small second category, ``_PARAMETERIZED_TOOLS``, supports the
    service-specific tools (``SERVICE_STATUS`` / ``SERVICE_LOGS``) that need
    a unit name. The executable and every other argument are still fixed;
    only the single unit-name argument is caller-supplied, and it is
    validated against a strict systemd-unit-name charset (``normalize_unit_name``)
    before it is ever placed in an argv list. It is still never interpolated
    into a shell string - ``subprocess.run(argv, shell=False, ...)`` treats
    it as one inert argv element, exactly like every other tool here.
  * Commands are executed with ``subprocess.run(argv, shell=False, ...)``.
    No shell is ever invoked, so shell metacharacters (`;`, `|`, `&&`, `$()`,
    backticks, ...) are inert even if they somehow ended up in output we
    later interpolate anywhere.
  * Every tool is read-only: process/service/log/disk/network *listing* or
    *status* commands only. Nothing here starts, stops, restarts, deletes,
    or writes anything on the host.
  * Every execution is wrapped in exception handling and a timeout. A
    missing binary, a permissions error, a timeout, or any other failure is
    captured into a structured ``ToolResult`` and returned - never raised
    up to the caller and never left to crash the request.
  * Every result is JSON-serializable (`ToolResult.to_dict()`), including a
    parsed/structured view of the output so the LLM (and the frontend) get
    real fields to reason about, not just a wall of raw text.

This module knows nothing about intents, prompts, or the LLM - it only
knows how to safely run one of a fixed set of commands and structure what
comes back. See ``ops_intent_classifier.py`` for "which tool does this
question need" and ``ops_prompts.py`` / ``ops_assistant.py`` for how the
result is turned into an answer.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from app.config import get_settings
from app.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ToolExecutionError(Exception):
    """Raised only for programmer errors (e.g. unknown tool name).

    Actual host/command failures (timeout, missing binary, non-zero exit,
    invalid unit-name argument, ...) are never raised - they come back as a
    ``ToolResult`` with ``success=False`` so callers always get a JSON-able
    result.
    """


class ToolName(str, Enum):
    """The fixed, whitelisted set of tools this copilot may execute.

    This enum IS the whitelist. Nothing outside `_TOOLS` /
    `_PARAMETERIZED_TOOLS` (keyed by these members) can ever be run - there
    is no way to construct a tool call from an arbitrary string.
    """

    CPU_TOP = "cpu_top"                    # ps -eo pid,comm,%cpu --sort=-%cpu | head -6
    MEMORY_TOP = "memory_top"              # ps -eo pid,comm,%mem --sort=-%mem | head -6
    DISK_USAGE = "disk_usage"              # df -h
    SERVICES_RUNNING = "services_running"  # systemctl list-units --type=service --state=running
    SERVICES_FAILED = "services_failed"    # systemctl --failed
    LOGS_RECENT = "logs_recent"            # journalctl -n 50 --no-pager
    LOGS_ERROR = "logs_error"              # journalctl -p err -n 50 --no-pager
    NETWORK_PORTS = "network_ports"        # ss -tuln

    # --- Service-level intelligence (added) ---------------------------------
    SERVICES_ALL = "services_all"          # systemctl list-units --type=service --all
    SERVICE_STATUS = "service_status"      # systemctl show <unit> -p ... (parameterized)
    SERVICE_LOGS = "service_logs"          # journalctl -u <unit> -n 50 (parameterized)


@dataclass
class ToolResult:
    """Structured, JSON-safe result of running one whitelisted tool."""

    tool: str
    display_command: str
    success: bool
    exit_code: int | None
    duration_seconds: float
    parsed: Any                     # structured data - list[dict] / dict, LLM + UI friendly
    raw_output: str                 # raw stdout (truncated), kept for transparency/debugging
    row_count: int | None = None
    truncated: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "command": self.display_command,
            "success": self.success,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "parsed": self.parsed,
            "raw_output": self.raw_output,
            "error": self.error,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Output parsers - turn raw stdout into structured, LLM/UI-friendly data.
# Each parser is defensive: malformed/unexpected lines are skipped rather
# than raising, since a parsing hiccup should never take down the tool call.
# --------------------------------------------------------------------------

def _parse_ps_table(stdout: str, value_key: str, limit: int) -> list[dict]:
    """Parse `ps -eo pid,comm,%cpu|%mem --sort=-...` output into rows.

    Equivalent to piping through `head -(limit+1)` (header + limit rows),
    done in Python so the tool never needs a shell pipe.
    """
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    # lines[0] is the header ("PID COMMAND %CPU" / "PID COMMAND %MEM")
    for line in lines[1 : limit + 1]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_str, command, value_str = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            value = float(value_str)
        except ValueError:
            value = 0.0
        rows.append({"pid": pid, "process": command, value_key: value})
    return rows


def _parse_df(stdout: str) -> list[dict]:
    """Parse `df -h` output into per-filesystem rows."""
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    for line in lines[1:]:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        filesystem, size, used, avail, use_percent, mounted_on = parts
        try:
            use_percent_value = float(use_percent.rstrip("%"))
        except ValueError:
            use_percent_value = None
        rows.append(
            {
                "filesystem": filesystem,
                "size": size,
                "used": used,
                "available": avail,
                "use_percent": use_percent_value,
                "mounted_on": mounted_on,
            }
        )
    rows.sort(key=lambda r: (r["use_percent"] is None, -(r["use_percent"] or 0)))
    return rows


_UNIT_LINE_RE = re.compile(
    r"^\s*(?P<unit>\S+\.service)\s+(?P<load>\S+)\s+(?P<active>\S+)\s+(?P<sub>\S+)\s*(?P<description>.*)$"
)


def _parse_systemctl_units(stdout: str) -> list[dict]:
    """Parse `systemctl list-units --type=service ...` / `--failed` tables.

    Skips the header row and the summary/legend lines systemctl prints
    after the table (blank line, "N loaded units listed.", ...).
    """
    rows: list[dict] = []
    for line in stdout.splitlines():
        match = _UNIT_LINE_RE.match(line)
        if not match:
            continue
        rows.append(
            {
                "unit": match.group("unit"),
                "load": match.group("load"),
                "active": match.group("active"),
                "sub": match.group("sub"),
                "description": match.group("description").strip(),
            }
        )
    return rows


def _parse_journal_lines(stdout: str) -> list[str]:
    """journalctl output is already one log entry per line - just clean it up."""
    return [line.strip() for line in stdout.splitlines() if line.strip()]


_SS_HEADER_AND_STATE = re.compile(r"^\S+")


def _parse_ss(stdout: str) -> list[dict]:
    """Parse `ss -tuln` into rows of {protocol, state, local_address}."""
    lines = [l for l in stdout.splitlines() if l.strip()]
    if not lines:
        return []
    rows: list[dict] = []
    for line in lines[1:]:  # first line is the header (Netid State Recv-Q ...)
        parts = line.split()
        if len(parts) < 5:
            continue
        netid, state, recv_q, send_q, local_address = parts[0], parts[1], parts[2], parts[3], parts[4]
        peer_address = parts[5] if len(parts) > 5 else ""
        rows.append(
            {
                "protocol": netid,
                "state": state,
                "local_address": local_address,
                "peer_address": peer_address,
            }
        )
    return rows


def _parse_systemctl_show(stdout: str) -> dict:
    """Parse `systemctl show <unit> -p ...` (machine-readable `Key=Value` per
    line) into a compact, friendly structure for one specific service.

    This is deliberately used instead of parsing free-text `systemctl
    status` output: `show` is stable, scriptable key=value output, so there
    is nothing ambiguous to mis-parse.
    """
    raw: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        raw[key.strip()] = value.strip()

    load_state = raw.get("LoadState") or "unknown"
    active_state = raw.get("ActiveState") or "unknown"
    sub_state = raw.get("SubState") or "unknown"
    unit_file_state = raw.get("UnitFileState") or "unknown"

    main_pid_raw = raw.get("MainPID")
    main_pid: int | None
    try:
        main_pid = int(main_pid_raw) if main_pid_raw is not None else None
    except ValueError:
        main_pid = None
    if main_pid == 0:
        # systemd reports 0 for "no main process" (not running) - normalize to None.
        main_pid = None

    return {
        "unit": raw.get("Id") or "",
        "description": raw.get("Description") or "",
        "load_state": load_state,
        "active_state": active_state,
        "sub_state": sub_state,
        "unit_file_state": unit_file_state,
        "enabled": unit_file_state == "enabled",
        "main_pid": main_pid,
        "active_since": raw.get("ActiveEnterTimestamp") or None,
        "inactive_since": raw.get("InactiveEnterTimestamp") or None,
        "result": raw.get("Result") or None,
        # `systemd` reports LoadState "not-found" for a unit name that does
        # not correspond to any installed unit file on this host.
        "found": load_state not in ("not-found", "", "unknown"),
    }


# --------------------------------------------------------------------------
# Service unit-name validation - the ONLY user-influenced value that ever
# reaches an argv list. Validated against the systemd unit-name charset
# before it is used anywhere, independent of the shell=False protection.
# --------------------------------------------------------------------------

_SERVICE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-@:\\]{1,128}$")
_UNIT_SUFFIXES = (".service", ".socket", ".target", ".timer", ".mount", ".device", ".path")


def normalize_unit_name(name: str) -> str | None:
    """Validate + normalize a caller-supplied service hint into a systemd
    unit name, or return ``None`` if it doesn't look like a safe unit name.

    Only characters systemd itself allows in unit names are accepted
    (letters, digits, ``_.-@:``). Anything else (spaces, shell
    metacharacters, path separators, quotes, ...) is rejected outright
    rather than sanitized, since this value may end up as a single argv
    element passed to `systemctl`/`journalctl`.
    """
    if not name:
        return None
    candidate = name.strip().strip(".")
    if not candidate:
        return None
    if not _SERVICE_NAME_RE.match(candidate):
        return None
    if not candidate.lower().endswith(_UNIT_SUFFIXES):
        candidate = f"{candidate}.service"
    return candidate


# --------------------------------------------------------------------------
# Tool registry - the single source of truth for what may be executed.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _ToolSpec:
    argv: tuple[str, ...]          # fixed argv - NEVER built from user input
    description: str
    parser: Callable[[str], Any]
    timeout_seconds: int = 10


@dataclass(frozen=True)
class _ParamToolSpec:
    """Like `_ToolSpec`, but the argv is built from one validated unit name.

    Everything except that single argument is still a fixed literal - only
    `build_argv`'s input (already passed through `normalize_unit_name`) is
    caller-influenced, and it is still just one inert argv element under
    `shell=False`.
    """

    build_argv: Callable[[str], tuple[str, ...]]
    description: str
    parser: Callable[[str], Any]
    timeout_seconds: int = 10


_TOOLS: dict[ToolName, _ToolSpec] = {
    ToolName.CPU_TOP: _ToolSpec(
        argv=("ps", "-eo", "pid,comm,%cpu", "--sort=-%cpu"),
        description="Top CPU-consuming processes (equivalent to `ps -eo pid,comm,%cpu --sort=-%cpu | head -6`)",
        parser=lambda out: _parse_ps_table(out, "cpu_percent", limit=5),
    ),
    ToolName.MEMORY_TOP: _ToolSpec(
        argv=("ps", "-eo", "pid,comm,%mem", "--sort=-%mem"),
        description="Top memory-consuming processes (equivalent to `ps -eo pid,comm,%mem --sort=-%mem | head -6`)",
        parser=lambda out: _parse_ps_table(out, "memory_percent", limit=5),
    ),
    ToolName.DISK_USAGE: _ToolSpec(
        argv=("df", "-h"),
        description="Disk usage per mounted filesystem (`df -h`)",
        parser=_parse_df,
    ),
    ToolName.SERVICES_RUNNING: _ToolSpec(
        argv=("systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend"),
        description="Currently running systemd services (`systemctl list-units --type=service --state=running`)",
        parser=_parse_systemctl_units,
        timeout_seconds=15,
    ),
    ToolName.SERVICES_FAILED: _ToolSpec(
        argv=("systemctl", "--failed", "--no-pager", "--no-legend"),
        description="Failed systemd services (`systemctl --failed`)",
        parser=_parse_systemctl_units,
        timeout_seconds=15,
    ),
    ToolName.LOGS_RECENT: _ToolSpec(
        argv=("journalctl", "-n", "50", "--no-pager"),
        description="Most recent 50 system journal entries (`journalctl -n 50 --no-pager`)",
        parser=_parse_journal_lines,
        timeout_seconds=15,
    ),
    ToolName.LOGS_ERROR: _ToolSpec(
        argv=("journalctl", "-p", "err", "-n", "50", "--no-pager"),
        description="Most recent 50 error-level journal entries (`journalctl -p err -n 50 --no-pager`)",
        parser=_parse_journal_lines,
        timeout_seconds=15,
    ),
    ToolName.NETWORK_PORTS: _ToolSpec(
        argv=("ss", "-tuln"),
        description="Listening TCP/UDP sockets (`ss -tuln`)",
        parser=_parse_ss,
    ),
    ToolName.SERVICES_ALL: _ToolSpec(
        argv=("systemctl", "list-units", "--type=service", "--all", "--no-pager", "--no-legend"),
        description="All known systemd services, running or not (`systemctl list-units --type=service --all`)",
        parser=_parse_systemctl_units,
        timeout_seconds=15,
    ),
}


_PARAMETERIZED_TOOLS: dict[ToolName, _ParamToolSpec] = {
    ToolName.SERVICE_STATUS: _ParamToolSpec(
        build_argv=lambda unit: (
            "systemctl",
            "show",
            unit,
            "--no-pager",
            "-p",
            "Id,LoadState,ActiveState,SubState,UnitFileState,Description,MainPID,"
            "Result,ActiveEnterTimestamp,InactiveEnterTimestamp",
        ),
        description="Structured live status for one specific systemd service (`systemctl show <service>`)",
        parser=_parse_systemctl_show,
        timeout_seconds=10,
    ),
    ToolName.SERVICE_LOGS: _ParamToolSpec(
        build_argv=lambda unit: ("journalctl", "-u", unit, "-n", "50", "--no-pager"),
        description="Most recent 50 journal entries for one specific systemd service (`journalctl -u <service> -n 50`)",
        parser=_parse_journal_lines,
        timeout_seconds=15,
    ),
}


def list_tools() -> list[dict]:
    """Describe every whitelisted tool - used by the /api/assistant/tools endpoint."""
    fixed = [
        {
            "tool": name.value,
            "command": " ".join(spec.argv),
            "description": spec.description,
        }
        for name, spec in _TOOLS.items()
    ]
    parameterized = [
        {
            "tool": name.value,
            "command": " ".join(spec.build_argv("<service-name>")),
            "description": spec.description,
        }
        for name, spec in _PARAMETERIZED_TOOLS.items()
    ]
    return fixed + parameterized


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n[... {omitted} more characters truncated ...]", True


def _run_argv(
    tool_value: str,
    display_command: str,
    argv: tuple[str, ...],
    parser: Callable[[str], Any],
    timeout_seconds: int,
) -> ToolResult:
    """Shared, paranoid execution core used by every tool (fixed or
    parameterized). Never raises for host-level failures - always returns a
    structured `ToolResult`, success or failure.
    """
    max_chars = settings.execution_max_output_chars
    started = time.monotonic()

    try:
        result = subprocess.run(
            list(argv),               # fixed/validated argv, never shell=True, never user-built as a string
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,  # never block on an interactive prompt
        )
        duration = round(time.monotonic() - started, 3)
        stdout, truncated = _truncate(result.stdout or "", max_chars)

        warnings: list[str] = []
        if result.returncode != 0:
            # Non-zero exit isn't necessarily fatal for these tools (e.g.
            # `systemctl --failed` exits non-zero when failed units exist -
            # that's the answer, not an error; `systemctl show` on an
            # unknown unit exits 0 with LoadState=not-found). Surface it as
            # a warning with stderr attached rather than a hard failure.
            stderr_snippet = (result.stderr or "").strip()
            if stderr_snippet:
                warnings.append(f"Command exited with code {result.returncode}: {stderr_snippet}")

        try:
            parsed = parser(stdout)
            row_count = len(parsed) if isinstance(parsed, list) else None
        except Exception as exc:  # noqa: BLE001 - parsing must never break the tool call
            logger.warning("Parser for tool '%s' failed: %s", tool_value, exc)
            parsed = None
            row_count = None
            warnings.append(f"Could not parse command output into structured data: {exc}")

        return ToolResult(
            tool=tool_value,
            display_command=display_command,
            success=True,
            exit_code=result.returncode,
            duration_seconds=duration,
            parsed=parsed,
            raw_output=stdout,
            row_count=row_count,
            truncated=truncated,
            error=None,
            warnings=warnings,
        )

    except subprocess.TimeoutExpired as exc:
        duration = round(time.monotonic() - started, 3)
        logger.warning("Tool '%s' timed out after %ss", tool_value, timeout_seconds)
        return ToolResult(
            tool=tool_value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=duration,
            parsed=None,
            raw_output=(exc.stdout or ""),
            row_count=None,
            truncated=False,
            error=f"Timed out after {timeout_seconds} seconds",
        )
    except FileNotFoundError as exc:
        logger.error("Tool '%s' binary not found: %s", tool_value, exc)
        return ToolResult(
            tool=tool_value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=f"Command not available on this host: {exc}",
        )
    except PermissionError as exc:
        logger.error("Tool '%s' permission denied: %s", tool_value, exc)
        return ToolResult(
            tool=tool_value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=f"Permission denied running this command: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - last-resort safety net, never let a tool crash the request
        logger.error("Unexpected error running tool '%s': %s", tool_value, exc)
        return ToolResult(
            tool=tool_value,
            display_command=display_command,
            success=False,
            exit_code=None,
            duration_seconds=round(time.monotonic() - started, 3),
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error=str(exc),
        )


def execute_tool(tool_name: ToolName | str) -> ToolResult:
    """Run exactly one whitelisted, read-only, fixed-argv tool.

    Never raises for host-level failures (timeout, missing binary, non-zero
    exit, permission denied, ...) - those all come back as
    ``ToolResult(success=False, error=...)``. Only raises ``ToolExecutionError``
    if ``tool_name`` is not a recognized, whitelisted, fixed-argv tool - i.e.
    a programmer error, not something a user request could ever trigger.
    """
    try:
        tool = ToolName(tool_name)
    except ValueError as exc:
        raise ToolExecutionError(f"Unknown tool: {tool_name!r} is not a whitelisted tool") from exc

    if tool not in _TOOLS:
        raise ToolExecutionError(
            f"'{tool.value}' is a parameterized tool - use execute_service_tool() instead"
        )

    spec = _TOOLS[tool]
    display_command = " ".join(spec.argv)
    return _run_argv(tool.value, display_command, spec.argv, spec.parser, spec.timeout_seconds)


def execute_service_tool(tool_name: ToolName | str, service_name: str) -> ToolResult:
    """Run one of the parameterized, service-specific tools
    (``SERVICE_STATUS`` / ``SERVICE_LOGS``) against a single validated unit
    name.

    ``service_name`` is passed through ``normalize_unit_name`` first; if it
    doesn't look like a safe systemd unit name, NOTHING is executed and a
    failed ``ToolResult`` is returned instead - the same "never raise, never
    fabricate" contract as ``execute_tool``, just with one extra guard in
    front of the (still fixed, still ``shell=False``) argv construction.
    """
    try:
        tool = ToolName(tool_name)
    except ValueError as exc:
        raise ToolExecutionError(f"Unknown tool: {tool_name!r} is not a whitelisted tool") from exc

    if tool not in _PARAMETERIZED_TOOLS:
        raise ToolExecutionError(
            f"'{tool.value}' is not a parameterized/service tool - use execute_tool() instead"
        )

    spec = _PARAMETERIZED_TOOLS[tool]
    normalized = normalize_unit_name(service_name)
    if normalized is None:
        logger.warning("Rejected invalid service name for tool '%s': %r", tool.value, service_name)
        return ToolResult(
            tool=tool.value,
            display_command=f"<rejected invalid service name: {service_name!r}>",
            success=False,
            exit_code=None,
            duration_seconds=0.0,
            parsed=None,
            raw_output="",
            row_count=None,
            truncated=False,
            error="Service name failed validation and was never executed.",
        )

    argv = spec.build_argv(normalized)
    display_command = " ".join(argv)
    return _run_argv(tool.value, display_command, argv, spec.parser, spec.timeout_seconds)


def execute_tools(tool_names: list[ToolName | str]) -> dict[str, ToolResult]:
    """Run several whitelisted, fixed-argv tools and return their results
    keyed by tool name.

    Used when a single question needs more than one data source (e.g. "why
    is my system slow?" -> CPU_TOP + MEMORY_TOP). Each tool is executed and
    isolated the same way as ``execute_tool`` - one tool failing never
    prevents the others from running. Parameterized service tools are not
    included here - see ``execute_service_tool``.
    """
    results: dict[str, ToolResult] = {}
    for name in tool_names:
        tool = ToolName(name)
        results[tool.value] = execute_tool(tool)
    return results
