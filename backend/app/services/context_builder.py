"""
Context builder for the AI Ops Assistant.

Given a classified intent, gathers just the system data relevant to it
(rather than dumping the entire system-info payload into every prompt).
This keeps prompts small, fast, and focused - and keeps the mapping from
"kind of question" to "data we fetch" in one obvious place.
"""

from app.logger import get_logger
from app.services import docker_monitor, file_search, system_monitor
from app.services.intent_classifier import Intent

logger = get_logger(__name__)


def _safe(label: str, fn, *args, **kwargs):
    """Run a collector defensively; never let one failure break the context."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Context collector '%s' failed: %s", label, exc)
        return {"error": f"Could not collect {label}: {exc}"}


def _slim_process(p: dict) -> dict:
    """Keep only the fields the LLM actually needs to reason about a process.

    The full process dict (cmdline, created_at, num_threads, ...) is still
    what /api/system/processes returns to the frontend unchanged - this
    trimming only affects what gets serialized into the LLM prompt, to keep
    prompt tokens down for the CPU demo.
    """
    return {
        "pid": p.get("pid"),
        "name": p.get("name"),
        "cpu_percent": p.get("cpu_percent"),
        "memory_percent": p.get("memory_percent"),
    }


def build_context(intent: Intent, message: str) -> dict:
    """Gather the system data most relevant to the given intent."""
    if intent == Intent.PERFORMANCE:
        top_cpu = _safe(
            "top_processes", system_monitor.get_processes, limit=5, sort_by="cpu_percent"
        ).get("processes", [])
        top_mem = _safe(
            "top_processes", system_monitor.get_processes, limit=5, sort_by="memory_percent"
        ).get("processes", [])
        return {
            "cpu": _safe("cpu", system_monitor.get_cpu_info),
            "memory": _safe("memory", system_monitor.get_memory_info),
            "top_processes_by_cpu": [_slim_process(p) for p in top_cpu],
            "top_processes_by_memory": [_slim_process(p) for p in top_mem],
            "uptime_seconds": _safe("uptime", system_monitor.get_uptime_seconds),
        }

    if intent == Intent.SERVICE_MANAGEMENT:
        services_data = _safe("services", system_monitor.get_services, limit=200)
        target_services = _extract_mentioned_services(message, services_data.get("services", []))
        return {
            "matched_services": target_services,
            "total_services_found": services_data.get("total_services", 0),
            "collector_errors": services_data.get("errors", []),
        }

    if intent == Intent.DOCKER:
        return {"docker": _safe("docker", docker_monitor.get_docker_containers)}

    if intent == Intent.FILE_SEARCH:
        return {
            "large_files": _safe("large_files", file_search.find_large_files, min_size_mb=100, limit=10),
            "disk_usage": _safe("disk", system_monitor.get_disk_info),
        }

    if intent == Intent.LOG_ANALYSIS:
        logs_data = _safe("logs", system_monitor.get_recent_logs, lines=20, priority="warning")
        return {
            "recent_warning_and_error_logs": logs_data.get("entries", []),
            "collector_errors": logs_data.get("errors", []),
        }

    if intent == Intent.NETWORK:
        return {"network": _safe("network", system_monitor.get_network_info)}

    if intent == Intent.USERS_SESSIONS:
        return {"logged_in_users": _safe("users", system_monitor.get_logged_in_users)}

    # GENERAL fallback: a light-touch snapshot so the model still has some
    # grounding, without the cost of a full deep collection.
    return {
        "version": _safe("version", system_monitor.get_system_version),
        "uptime_seconds": _safe("uptime", system_monitor.get_uptime_seconds),
        "cpu_usage_percent": _safe("cpu", system_monitor.get_cpu_info).get("usage_percent"),
        "memory_usage_percent": _safe("memory", system_monitor.get_memory_info).get("usage_percent"),
    }


def _extract_mentioned_services(message: str, services: list[dict]) -> list[dict]:
    """Best-effort match of service names mentioned in the user's message.

    Falls back to returning nothing (rather than the whole service list) if
    no name matches, so the LLM is nudged to ask a clarifying question
    instead of guessing which service the user means.
    """
    text = message.lower()
    matched = []
    for service in services:
        name = service.get("name", "")
        base_name = name.replace(".service", "").lower()
        if base_name and base_name in text:
            matched.append(service)
    return matched