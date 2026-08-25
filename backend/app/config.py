"""
Configuration module.

Loads settings from environment variables / .env file using pydantic-settings.
Keep this simple - a single Settings object used across the app.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "Linux Copilot XAI"
    app_env: str = "development"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./linux_copilot.db"

    # CORS - comma separated string, parsed into a list.
    # Includes every port this project's own docs tell someone to serve the
    # frontend on: python3 -m http.server default doc'd port (5500), the
    # common Vite/CRA dev ports (5173/3000), and 8000/8080 in case the
    # frontend is ever served from the same box as the API. Widen further
    # (or set to "*") via the CORS_ORIGINS env var if your demo setup uses
    # a different host/port.
    cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5500,http://127.0.0.1:5500,"
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8080,http://127.0.0.1:8080"
    )

    # Logging
    log_level: str = "INFO"

    # Ollama (local LLM) - used by the AI Ops Assistant
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 140  # read timeout; was 60 - CPU demo target is a 5-10s reply
    ollama_connect_timeout_seconds: int = 5  # fail fast if Ollama isn't up at all
    ollama_temperature: float = 0.2

    # CPU-demo speed tuning (see ollama_client.chat):
    ollama_think: bool = False  # disable "thinking" traces on reasoning-capable models (e.g. Qwen3)
    ollama_num_predict: int = 400  # cap generated tokens - long tail of a rambling reply is the #1 latency cost
    ollama_num_ctx: int = 2048  # smaller context window = faster prompt processing on CPU
    ollama_keep_alive: str = "30m"  # keep the model loaded between requests instead of reloading each call

    # AI Assistant behavior
    assistant_history_turns: int = 2  # number of past turns fed back into the prompt
    assistant_max_history_stored: int = 50  # per session, in the DB

    # Safe Command Execution behavior
    execution_timeout_seconds: int = 30  # max seconds a confirmed command may run before being killed
    # systemctl/service start|restart|reload|try-restart commands legitimately take
    # longer than a normal command: e.g. NetworkManager-wait-online.service is a
    # oneshot unit that blocks on `nm-online` for up to ~60s, and `systemctl
    # restart` waits for that ExecStart to finish before returning. These get a
    # longer budget instead of being killed as if they'd hung.
    service_restart_timeout_seconds: int = 90
    execution_max_output_chars: int = 20000  # stdout/stderr truncated beyond this, per stream
    execution_max_history_stored: int = 100  # command executions kept per session, in the DB

    # --- Shared system thresholds ---
    # Used by the AI One-Click Fix Engine (app/services/fix_engine.py) for
    # its High CPU / High Memory / Disk Almost Full detectors.
    alert_cpu_threshold_percent: float = 90.0
    alert_memory_threshold_percent: float = 90.0
    alert_disk_threshold_percent: float = 90.0

    # --- Sprint 8: AI One-Click Fix Engine ---
    fix_apache_service_name: str = "apache2"  # systemd unit basename checked by the "Apache Down" detector

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a clean list of strings."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def sqlite_db_path(self) -> str:
        """Extract the filesystem path from the sqlite DATABASE_URL."""
        return self.database_url.replace("sqlite:///", "", 1)


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance so we don't re-read the .env file every call."""
    return Settings()