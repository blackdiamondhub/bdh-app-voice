"""The app's settings (Settings → Voice), read with a short cache.

In the container ``app_setting`` asks the hub broker, so the listen socket
re-reads them at most every :data:`TTL` seconds rather than per frame. The
assistant's API key is a secret and is read separately (:func:`api_key`).
"""
import time
from dataclasses import dataclass

from bdh_sdk import secrets
from bdh_sdk.settings import app_setting, as_bool

from .detector import DEFAULT_WORD, WAKE_WORDS

APP_ID = "voice"
TTL = 10.0
DEFAULT_THRESHOLD = 50  # percent

DEFAULT_ENDPOINT = "https://bdh-ai-lodge.openai.azure.com"
MODELS = (("gpt-realtime-2.1-mini", "gpt-realtime-2.1-mini (fast, cheaper)"),
          ("gpt-realtime-2.1", "gpt-realtime-2.1 (smarter)"))
DEFAULT_MODEL = MODELS[0][0]
VOICES = ("marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")
DEFAULT_VOICE = "marin"
DEFAULT_MAX_SECONDS = 90
DEFAULT_FOLLOW_UP = 8


@dataclass(frozen=True)
class Config:
    listen: bool = True
    word: str = DEFAULT_WORD
    threshold: float = DEFAULT_THRESHOLD / 100
    save_clips: bool = True
    # The voice assistant (a conversation after each wake word).
    assistant: bool = True
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    max_seconds: float = DEFAULT_MAX_SECONDS
    follow_up_seconds: float = DEFAULT_FOLLOW_UP
    instructions: str = ""
    # "<screen ip or name>=<room>; ..." or just "<room>" for every screen.
    screen_rooms: str = ""


def room_for(cfg, *screen_ids):
    """The room a screen is in, from the "Where the screens are" setting, or ""."""
    fallback = ""
    for part in cfg.screen_rooms.replace("\n", ";").split(";"):
        key, sep, room = part.partition("=")
        if not sep:
            fallback = fallback or key.strip()
        elif key.strip() and key.strip() in screen_ids:
            return room.strip()
    return fallback


def _bool(key, default):
    try:
        return as_bool(app_setting(APP_ID, key, default))
    except ValueError:
        return default


def _int(key, default, low, high):
    try:
        value = int(app_setting(APP_ID, key, default))
    except (TypeError, ValueError):
        value = default
    return min(max(value, low), high)


def _choice(key, default, allowed):
    value = str(app_setting(APP_ID, key, default) or default)
    return value if value in allowed else default


def load():
    pct = _int("threshold", DEFAULT_THRESHOLD, 1, 99)
    return Config(
        listen=_bool("listen", True),
        word=_choice("wake_word", DEFAULT_WORD, WAKE_WORDS),
        threshold=pct / 100,
        save_clips=_bool("save_clips", True),
        assistant=_bool("enabled", True),
        endpoint=str(app_setting(APP_ID, "endpoint", DEFAULT_ENDPOINT) or DEFAULT_ENDPOINT).strip(),
        model=_choice("model", DEFAULT_MODEL, [m for m, _ in MODELS]),
        voice=_choice("voice", DEFAULT_VOICE, VOICES),
        max_seconds=_int("max_seconds", DEFAULT_MAX_SECONDS, 10, 600),
        follow_up_seconds=_int("follow_up_seconds", DEFAULT_FOLLOW_UP, 2, 60),
        instructions=str(app_setting(APP_ID, "instructions", "") or "").strip(),
        screen_rooms=str(app_setting(APP_ID, "screen_rooms", "") or "").strip(),
    )


def api_key():
    """The realtime API key (Settings → Voice → API key), or "" if unset."""
    try:
        return secrets.get("api_key", "", app=APP_ID) or ""
    except Exception:  # the broker is unreachable: treat as unset, the wake still chimes
        return ""


_cache = (0.0, None)


def current(max_age=TTL):
    global _cache
    at, cfg = _cache
    now = time.monotonic()
    if cfg is None or now - at > max_age:
        cfg = load()
        _cache = (now, cfg)
    return cfg


def forget():
    global _cache
    _cache = (0.0, None)
