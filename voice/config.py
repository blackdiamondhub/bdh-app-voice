"""The app's settings (Settings → Voice), read with a short cache.

In the container ``app_setting`` asks the hub broker, so the listen socket
re-reads them at most every :data:`TTL` seconds rather than per frame.
"""
import time
from dataclasses import dataclass

from bdh_sdk.settings import app_setting, as_bool

from .detector import DEFAULT_WORD, WAKE_WORDS

APP_ID = "voice"
TTL = 10.0
DEFAULT_THRESHOLD = 50  # percent


@dataclass(frozen=True)
class Config:
    listen: bool = True
    word: str = DEFAULT_WORD
    threshold: float = DEFAULT_THRESHOLD / 100
    save_clips: bool = True


def _bool(key, default):
    try:
        return as_bool(app_setting(APP_ID, key, default))
    except ValueError:
        return default


def load():
    word = str(app_setting(APP_ID, "wake_word", DEFAULT_WORD) or DEFAULT_WORD)
    if word not in WAKE_WORDS:
        word = DEFAULT_WORD
    try:
        pct = int(app_setting(APP_ID, "threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        pct = DEFAULT_THRESHOLD
    pct = min(max(pct, 1), 99)
    return Config(listen=_bool("listen", True), word=word, threshold=pct / 100,
                  save_clips=_bool("save_clips", True))


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
