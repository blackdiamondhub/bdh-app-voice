"""Audio around each detection, and the detection log.

Each connection keeps its recent audio. After a wake it waits for
:data:`AFTER_SECONDS` more, then saves 2 s before + 1 s after the wake as a
WAV under the app's data directory (``/data/voice`` in the
container). Only the newest :data:`KEEP_CLIPS` clips are kept; the log keeps
the newest :data:`KEEP_DETECTIONS` detections.

Clips are people talking in the lodge, so they live outside ``MEDIA_ROOT``
(which the runtime serves to anyone the hub sends) and are only served by the
admin-only view.
"""
import io
import json
import logging
import re
import threading
import wave
from pathlib import Path

import numpy as np

from .detector import SAMPLE_RATE

log = logging.getLogger(__name__)

BEFORE_SECONDS = 2
AFTER_SECONDS = 1
RING_SECONDS = BEFORE_SECONDS + AFTER_SECONDS
KEEP_CLIPS = 100
KEEP_DETECTIONS = 50
CLIP_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.wav$")

_lock = threading.Lock()


def data_dir():
    from django.conf import settings

    explicit = getattr(settings, "VOICE_DATA_DIR", None)
    if explicit:
        path = Path(explicit)
    elif getattr(settings, "DATA_DIR", None):
        path = Path(settings.DATA_DIR) / "voice"
    else:
        path = Path(settings.MEDIA_ROOT).parent / "voice-data"
    return path


def clips_dir():
    return data_dir() / "clips"


class Ring:
    """The newest ``seconds`` of int16 audio."""

    def __init__(self, seconds=RING_SECONDS, rate=SAMPLE_RATE):
        self.size = int(seconds * rate)
        self._buf = np.zeros(self.size, dtype=np.int16)
        self._filled = 0
        self._pos = 0

    def add(self, pcm):
        pcm = np.asarray(pcm, dtype=np.int16)[-self.size:]
        n = len(pcm)
        end = self._pos + n
        if end <= self.size:
            self._buf[self._pos:end] = pcm
        else:
            first = self.size - self._pos
            self._buf[self._pos:] = pcm[:first]
            self._buf[:n - first] = pcm[first:]
        self._pos = end % self.size
        self._filled = min(self.size, self._filled + n)

    def samples(self, count=None, skip=0):
        """Up to ``count`` samples ending ``skip`` samples before the newest."""
        if self._filled < self.size:
            data = self._buf[:self._filled].copy()
        else:
            data = np.concatenate((self._buf[self._pos:], self._buf[:self._pos]))
        if skip:
            data = data[:-skip] if skip < len(data) else data[:0]
        if count is not None:
            data = data[-count:] if count else data[:0]
        return data


class ClipRecorder:
    """Keeps recent audio and hands back each clip once its tail has arrived.

    Clips are placed by sample position, so audio that arrives in big
    batches (a detector catching up) still gives the right 2 s + 1 s.
    """

    SLACK_SECONDS = 6  # covers a batch of up to 5 s of backlog

    def __init__(self, before=BEFORE_SECONDS, after=AFTER_SECONDS, rate=SAMPLE_RATE):
        self.before = int(before * rate)
        self.after = int(after * rate)
        self.ring = Ring(before + after + self.SLACK_SECONDS, rate)
        self.total = 0
        self._waiting = []  # (end sample, tag)

    def add(self, pcm):
        """Add audio; return ``[(tag, samples)]`` for clips that are now complete."""
        self.ring.add(pcm)
        self.total += len(pcm)
        done = []
        while self._waiting and self._waiting[0][0] <= self.total:
            end, tag = self._waiting.pop(0)
            done.append((tag, self.ring.samples(self.before + self.after, self.total - end)))
        return done

    def mark(self, tag, at=None):
        """A wake happened at sample ``at`` (default: now): save the clip around it as ``tag``."""
        at = self.total if at is None else at
        self._waiting.append((at + self.after, tag))
        self._waiting.sort(key=lambda item: item[0])


def wav_bytes(samples, rate=SAMPLE_RATE):
    out = io.BytesIO()
    with wave.open(out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return out.getvalue()


def save_clip(name, samples, keep=KEEP_CLIPS):
    """Write ``name`` (a .wav file name) and delete all but the newest ``keep`` clips."""
    if not CLIP_NAME.match(name):
        raise ValueError(f"bad clip name {name!r}")
    folder = clips_dir()
    with _lock:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(wav_bytes(samples))
        clips = sorted(folder.glob("*.wav"), key=lambda p: (p.stat().st_mtime, p.name))
        for old in clips[:-keep] if keep else clips:
            old.unlink(missing_ok=True)
    return folder / name


def clip_path(name):
    """The saved clip called ``name``, or None."""
    if not name or not CLIP_NAME.match(name):
        return None
    path = clips_dir() / name
    return path if path.is_file() else None


def _log_path():
    return data_dir() / "detections.json"


def detections():
    """The detection log, newest first."""
    try:
        data = json.loads(_log_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record(entry, keep=KEEP_DETECTIONS):
    with _lock:
        items = [entry] + [d for d in detections() if d.get("id") != entry.get("id")]
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items[:keep]), encoding="utf-8")
        tmp.replace(path)
