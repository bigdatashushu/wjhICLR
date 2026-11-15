"""Local configuration, with lazy credential loading and no shell evaluation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigurationError

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILES = PACKAGE_DIR / "profiles.json"
DEFAULT_CREDENTIALS = PACKAGE_DIR / "credentials.local.env"
KEY_NAMES = frozenset({
    "XCODE_API_KEY", "XCODE_GPT_API_KEY",
    "XCODE_CLAUDE_API_KEY", "XCODE_GEMINI_API_KEY",
})
PROTOCOL_PATHS = {
    "responses": "/responses",
    "messages": "/messages",
    "chat_completions": "/chat/completions",
}


@dataclass(frozen=True)
class Profile:
    protocol: str
    base_url: str
    model: str
    api_key_env: str
    auth: str = "bearer"
    timeout_seconds: float = 180

    def __post_init__(self):
        if self.protocol not in PROTOCOL_PATHS:
            raise ConfigurationError("Unknown protocol.")
        if not isinstance(self.base_url, str):
            raise ConfigurationError("base_url must be a string.")
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except ValueError:
            raise ConfigurationError("Invalid base_url.") from None
        if (
            parsed.scheme != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or (port is not None and port <= 0)
            or any(c.isspace() for c in self.base_url)
        ):
            raise ConfigurationError("base_url must be an HTTPS API base without credentials or query.")
        if parsed.path.rstrip("/") not in ("", "/v1"):
            raise ConfigurationError("Use the site root or /v1 as base_url.")
        if not isinstance(self.model, str) or any(c.isspace() for c in self.model):
            raise ConfigurationError("model must be an exact model ID without whitespace.")
        if self.api_key_env not in KEY_NAMES:
            raise ConfigurationError("Use a documented XCODE_*_API_KEY variable.")
        if self.auth not in ("bearer", "api_key"):
            raise ConfigurationError("auth must be bearer or api_key.")
        if self.protocol != "messages" and self.auth != "bearer":
            raise ConfigurationError("OpenAI-compatible protocols require bearer authentication.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ConfigurationError("timeout_seconds must be positive and finite.")

    @property
    def api_base(self) -> str:
        base = self.base_url.rstrip("/")
        return base if urlsplit(base).path == "/v1" else base + "/v1"

    @property
    def endpoint(self) -> str:
        return self.api_base + PROTOCOL_PATHS[self.protocol]

    def with_model(self, model: str | None) -> Profile:
        return self if model is None else replace(self, model=model)

    def require_model(self):
        if not self.model:
            raise ConfigurationError("Set an exact model ID from your token group's model list.")


def load_profiles(path: str | Path | None = None) -> dict[str, Profile]:
    """Load public defaults, or an explicitly selected local configuration."""
    try:
        data = json.loads(Path(path or DEFAULT_PROFILES).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError
        allowed = {"protocol", "base_url", "model", "api_key_env", "auth", "timeout_seconds"}
        profiles = {}
        for name, fields in data.items():
            if not isinstance(name, str) or not isinstance(fields, dict) or set(fields) - allowed:
                raise ValueError
            profiles[name] = Profile(**fields)
        return profiles
    except (OSError, UnicodeError, ValueError, TypeError):
        raise ConfigurationError("Cannot load profiles; check JSON and documented fields.") from None


def read_credentials(path: str | Path = DEFAULT_CREDENTIALS) -> dict[str, str]:
    """Read literal KEY=value lines. Never source, interpolate, or print values."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        if os.name == "posix" and path.stat().st_mode & 0o077:
            raise ConfigurationError("Private credential file must have permissions 600.")
        result = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if not separator or name not in KEY_NAMES or name in result:
                raise ConfigurationError("Invalid or duplicate credential entry.")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if any(c.isspace() or ord(c) < 32 for c in value):
                raise ConfigurationError("API tokens cannot contain whitespace or control characters.")
            result[name] = value
        return result
    except (OSError, UnicodeError):
        raise ConfigurationError("Cannot read private credential file.") from None


def resolve_api_key(profile: Profile, path: str | Path = DEFAULT_CREDENTIALS) -> str:
    # Dedicated key wins, even when the common environment key is set.
    dedicated = os.environ.get(profile.api_key_env, "")
    if dedicated:
        key = dedicated
    else:
        local = read_credentials(path)
        key = local.get(profile.api_key_env) or os.environ.get("XCODE_API_KEY") or local.get("XCODE_API_KEY")
    if not key:
        raise ConfigurationError("Missing Xcode API token for this profile.")
    if any(c.isspace() or ord(c) < 32 for c in key):
        raise ConfigurationError("Invalid API token format.")
    return key
