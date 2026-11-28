"""Wake-word training clips: real audio from the screens, for a custom model.

Every time a screen's wake score peaks at or above the *collect threshold*
(lower than the wake threshold, so near-misses count too) the listen session
asks for a clip of :data:`BEFORE_SECONDS` before the peak to
:data:`AFTER_SECONDS` after it: 16 kHz mono 16-bit WAV, the format
openWakeWord trains on. :class:`Events` groups a run of high scores into one
event and takes its peak.

Clips are written by one background thread (:class:`Writer`) fed through a
bounded queue, so the listen loop never waits on the disk; when the queue is
full the clip is dropped. Each clip has a row in a small SQLite index next to
the WAV files: when and where, the scores, whether it woke the assistant and,
once that conversation ends, how it went (:func:`outcome`). From that comes
an automatic label suggestion; an admin can set the real one on the review page.

Everything lives under the app's data directory (``/data/voice/dataset`` in
the container), never in ``MEDIA_ROOT`` and never sent anywhere: only the
admin-only review page serves it.
"""
import io
import json
import logging
import queue
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone

from . import clips
from .config import DEFAULT_MAX_CLIPS
from .detector import SAMPLE_RATE

log = logging.getLogger(__name__)

BEFORE_SECONDS = 2.0
AFTER_SECONDS = 1.0
QUEUE_SIZE = 32
STAY_TTL = 60.0  # seconds a stays.current answer is trusted
TRANSCRIPT_CHARS = 200

NEAR_MISS = "near_miss"
LIKELY_TRUE = "likely_true"
LIKELY_FALSE = "likely_false"
UNKNOWN = "unknown"
AUTO_LABELS = (LIKELY_TRUE, LIKELY_FALSE, NEAR_MISS, UNKNOWN)
CLIP_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$")

_COLUMNS = (
    ("id", "TEXT PRIMARY KEY"), ("at", "TEXT"), ("ip", "TEXT"), ("screen", "TEXT"),
    ("word", "TEXT"), ("score", "REAL"), ("threshold", "REAL"), ("collect_threshold", "REAL"),
    ("fired", "INTEGER"), ("seconds", "REAL"), ("end_reason", "TEXT"),
    ("duration", "REAL"), ("user_turns", "INTEGER"), ("tool_calls", "INTEGER"),
    ("replies", "INTEGER"), ("silent_end", "INTEGER"), ("transcript", "TEXT"),
    ("auto_label", "TEXT"), ("label", "TEXT"), ("labelled_at", "TEXT"),
)
FIELDS = tuple(name for name, _ in _COLUMNS)

_db_lock = threading.RLock()


# --- where things live ------------------------------------------------------------

def root():
    return clips.data_dir() / "dataset"


def wav_dir():
    return root() / "clips"


def wav_path(clip_id):
    """The WAV of ``clip_id`` (whether or not it exists), or None for a bad id."""
    if not isinstance(clip_id, str) or not CLIP_ID.match(clip_id):
        return None
    return wav_dir() / f"{clip_id}.wav"


def _connect():
    path = root() / "index.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS clips (%s)" % ", ".join(f"{n} {t}" for n, t in _COLUMNS))
    db.execute("CREATE INDEX IF NOT EXISTS clips_at ON clips (at)")
    return db


def _query(sql, args=()):
    with _db_lock:
        db = _connect()
        try:
            with db:
                return [dict(r) for r in db.execute(sql, args)]
        finally:
            db.close()


def new_id(now=None):
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


# --- labels -------------------------------------------------------------------------

def auto_label(row):
    """The suggested label of an index row (a dict with the clip's fields)."""
    if not row.get("fired"):
        return NEAR_MISS
    if not row.get("end_reason"):
        return UNKNOWN
    turns = row.get("user_turns") or 0
    if row.get("silent_end"):
        return LIKELY_FALSE  # the model judged it chatter and ended without a word
    if turns == 0 and row.get("end_reason") == "idle":
        return LIKELY_FALSE
    if turns >= 1 and ((row.get("tool_calls") or 0) or (row.get("replies") or 0)):
        return LIKELY_TRUE
    return UNKNOWN


def folder(row):
    """``positive``, ``negative`` or ``unlabelled``: the human label, else the suggestion.

    Near-misses count as negatives; clips whose outcome is unknown stay unlabelled.
    """
    if row.get("label") == "true":
        return "positive"
    if row.get("label") == "false":
        return "negative"
    return {LIKELY_TRUE: "positive", LIKELY_FALSE: "negative",
            NEAR_MISS: "negative"}.get(row.get("auto_label"), "unlabelled")


def set_label(clip_id, label):
    """Set the human label (``"true"``, ``"false"`` or None to clear); False if no such clip."""
    if label not in ("true", "false", None):
        raise ValueError(f"bad label {label!r}")
    at = datetime.now(timezone.utc).isoformat() if label else None
    with _db_lock:
        db = _connect()
        try:
            with db:
                cur = db.execute("UPDATE clips SET label = ?, labelled_at = ? WHERE id = ?",
                                 (label, at, clip_id))
                return cur.rowcount > 0
        finally:
            db.close()


# --- reading --------------------------------------------------------------------------

def _where(label_filter):
    if label_filter in ("true", "false"):
        return "WHERE label = ?", (label_filter,)
    if label_filter == "unlabelled":
        return "WHERE label IS NULL", ()
    if label_filter in AUTO_LABELS:
        return "WHERE label IS NULL AND auto_label = ?", (label_filter,)
    return "", ()


def rows(label_filter="", limit=100, offset=0):
    """Index rows, newest first."""
    where, args = _where(label_filter)
    return _query(f"SELECT * FROM clips {where} ORDER BY at DESC, id DESC LIMIT ? OFFSET ?",
                  (*args, int(limit), int(offset)))


def count(label_filter=""):
    where, args = _where(label_filter)
    return _query(f"SELECT COUNT(*) AS n FROM clips {where}", args)[0]["n"]


def get(clip_id):
    found = _query("SELECT * FROM clips WHERE id = ?", (clip_id,))
    return found[0] if found else None


def counts():
    """Clips per human label and per suggestion (for unlabelled clips)."""
    out = {"all": 0, "true": 0, "false": 0, "unlabelled": 0}
    for r in _query("SELECT label, auto_label, COUNT(*) AS n FROM clips GROUP BY label, auto_label"):
        out["all"] += r["n"]
        if r["label"] in ("true", "false"):
            out[r["label"]] += r["n"]
        else:
            out["unlabelled"] += r["n"]
            out[r["auto_label"] or UNKNOWN] = out.get(r["auto_label"] or UNKNOWN, 0) + r["n"]
    return out


# --- writing ----------------------------------------------------------------------------

def _insert(meta):
    row = {k: meta.get(k) for k in FIELDS}
    row["auto_label"] = auto_label(row)
    with _db_lock:
        db = _connect()
        try:
            with db:
                db.execute("INSERT OR REPLACE INTO clips (%s) VALUES (%s)"
                           % (", ".join(FIELDS), ", ".join("?" * len(FIELDS))),
                           [row[k] for k in FIELDS])
        finally:
            db.close()


def _apply_outcome(clip_id, result):
    """Join a conversation's outcome to its clip; False if the clip isn't indexed (yet)."""
    with _db_lock:
        row = get(clip_id)
        if row is None:
            return False
        row.update(result)
        row["auto_label"] = auto_label(row)
        db = _connect()
        try:
            with db:
                db.execute("UPDATE clips SET end_reason = ?, duration = ?, user_turns = ?, "
                           "tool_calls = ?, replies = ?, silent_end = ?, transcript = ?, "
                           "auto_label = ? WHERE id = ?",
                           (row["end_reason"], row["duration"], row["user_turns"],
                            row["tool_calls"], row["replies"], row["silent_end"],
                            row["transcript"], row["auto_label"], clip_id))
        finally:
            db.close()
    return True


def prune(max_clips=DEFAULT_MAX_CLIPS):
    """Delete the oldest clips until ``max_clips`` remain; returns the number deleted.

    Unlabelled clips go first; labelled ones only when they alone are over the cap.
    """
    extra = count() - max_clips
    if extra <= 0:
        return 0
    doomed = [r["id"] for r in _query(
        "SELECT id FROM clips ORDER BY (label IS NOT NULL), at, id LIMIT ?", (extra,))]
    with _db_lock:
        db = _connect()
        try:
            with db:
                db.executemany("DELETE FROM clips WHERE id = ?", [(i,) for i in doomed])
        finally:
            db.close()
    for clip_id in doomed:
        path = wav_path(clip_id)
        if path:
            path.unlink(missing_ok=True)
    return len(doomed)


# --- guest stays --------------------------------------------------------------------

class StayCheck:
    """Is a guest (not an owner) staying? Asked of the hub at most every :data:`STAY_TTL` s.

    When the hub can't say (no ``stays.read`` grant, broker down) it keeps the
    last answer; with no answer yet it assumes a guest, so nothing is recorded.
    """

    def __init__(self, ttl=STAY_TTL):
        self.ttl = ttl
        self._at = None
        self._guest = None
        self._current = None  # the last CurrentStay the hub reported (None between stays)
        self._warned = False

    def _refresh(self):
        now = time.monotonic()
        if self._at is None or now - self._at >= self.ttl:
            self._at = now
            try:
                from bdh_sdk import stays

                current = stays.current()
                self._current = current
                self._guest = bool(current and not current.stay.owner)
                self._warned = False
            except Exception as exc:
                if not self._warned:
                    log.warning("can't tell whether guests are staying (%s)", exc)
                    self._warned = True

    def guest_staying(self):
        self._refresh()
        return True if self._guest is None else self._guest

    def info(self):
        """``{"stay", "stay_key", "stay_checkout"}``; ``stay`` is guest, owner, none or unknown."""
        self._refresh()
        if self._guest is None:
            return {"stay": "unknown", "stay_key": "", "stay_checkout": ""}
        current = self._current
        if current is None:
            return {"stay": "none", "stay_key": "", "stay_checkout": ""}
        checkout = current.stay.checkout
        return {"stay": "owner" if current.stay.owner else "guest", "stay_key": current.key or "",
                "stay_checkout": checkout.isoformat() if checkout else ""}


# --- the writer thread ----------------------------------------------------------------

class Writer:
    """Saves clips and outcomes in order, on its own thread."""

    PENDING_OUTCOMES = 50

    def __init__(self, settings=None, stays=None):
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread = None
        self._lock = threading.Lock()
        self._settings = settings
        self.stays = stays or StayCheck()
        self._outcomes = {}  # clip id -> outcome that arrived before its clip
        self.dropped = 0

    def _start(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="voice-dataset", daemon=True)
                self._thread.start()

    def _put(self, item):
        self._start()
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            log.warning("dataset writer is behind: dropped a clip (%d so far)", self.dropped)
            return False

    def clip(self, meta, samples):
        """Queue a clip (its index fields and int16 samples) for saving. Never blocks."""
        return self._put(("clip", meta, samples))

    def outcome(self, clip_id, result):
        return self._put(("outcome", clip_id, result))

    def flush(self, timeout=10):
        """Wait until everything queued so far is written (for tests)."""
        done = threading.Event()
        self._start()
        self._queue.put(("flush", done, None), timeout=timeout)
        return done.wait(timeout)

    def _cfg(self):
        if self._settings is not None:
            return self._settings()
        from . import config

        return config.current()

    def _run(self):
        while True:
            kind, a, b = self._queue.get()
            try:
                if kind == "flush":
                    a.set()
                elif kind == "clip":
                    self._save(a, b)
                elif kind == "outcome":
                    if not _apply_outcome(a, b):
                        self._outcomes[a] = b
                        while len(self._outcomes) > self.PENDING_OUTCOMES:
                            self._outcomes.pop(next(iter(self._outcomes)))
            except Exception:
                log.exception("dataset writer: could not save %s", kind)

    def _save(self, meta, samples):
        cfg = self._cfg()
        if not cfg.collect_clips:
            return
        if not cfg.collect_during_stays and self.stays.guest_staying():
            return
        path = wav_path(meta["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(clips.wav_bytes(samples))
        tmp.replace(path)
        meta = dict(meta, seconds=round(len(samples) / SAMPLE_RATE, 3))
        _insert(meta)
        pending = self._outcomes.pop(meta["id"], None)
        if pending:
            _apply_outcome(meta["id"], pending)
        prune(cfg.collect_max_clips)


writer = Writer()


def outcome(clip_id, *, reason, duration, user_turns, tool_calls, replies, silent_end,
            transcript=""):
    """The conversation a clip's wake started has ended: record how it went."""
    if not clip_id:
        return
    writer.outcome(clip_id, {
        "end_reason": reason, "duration": round(duration, 1), "user_turns": user_turns,
        "tool_calls": tool_calls, "replies": replies, "silent_end": int(bool(silent_end)),
        "transcript": (transcript or "")[:TRANSCRIPT_CHARS],
    })


# --- events (in the listen session) --------------------------------------------------

class Events:
    """Turns a screen's stream of scores into one clip request per detection event.

    An event starts at the first score at or above the collect threshold and
    ends at the first score below it, at a wake, or after :data:`MAX_SECONDS`;
    its clip is centred on the peak. A new event can't start within
    ``debounce`` seconds of the last peak, unless it wakes the assistant.
    """

    MAX_SECONDS = 2.0

    def __init__(self, debounce, rate=SAMPLE_RATE):
        self.debounce = int(debounce * rate)
        self.max_len = int(self.MAX_SECONDS * rate)
        self.open = None  # [start sample, peak score, peak sample, fired]
        self.last_peak = None

    def score(self, score, at, collect_threshold, fired=False):
        """One chunk's score at sample ``at``; return a finished event ``(peak, at, fired)`` or None."""
        done = None
        if self.open is not None:
            if score < collect_threshold or at - self.open[0] > self.max_len:
                done = self._close()
            else:
                if score > self.open[1]:
                    self.open[1], self.open[2] = score, at
                self.open[3] = self.open[3] or fired
                if fired:
                    return self._close()
                return None
        if score >= collect_threshold and (
                fired or self.last_peak is None or at - self.last_peak >= self.debounce):
            self.open = [at, score, at, fired]
            if fired:
                return self._close()
        return done

    def close(self):
        """End the open event (e.g. detection pauses for a conversation)."""
        return self._close() if self.open is not None else None

    def _close(self):
        _, peak, at, fired = self.open
        self.open = None
        self.last_peak = at
        return peak, at, fired


# --- export -----------------------------------------------------------------------------

class _Sink(io.RawIOBase):
    """A write-only stream that hands back what was written (for a streamed zip)."""

    def __init__(self):
        self.parts = []

    def writable(self):
        return True

    def write(self, b):
        self.parts.append(bytes(b))
        return len(b)

    def take(self):
        data, self.parts = b"".join(self.parts), []
        return data


def zip_stream():
    """Yield a zip of every clip: ``positive/``, ``negative/``, ``unlabelled/`` and ``metadata.jsonl``.

    Folders follow the human label, else the suggestion (near-misses are
    negatives, unknowns stay unlabelled). WAVs are stored, not compressed.
    """
    sink = _Sink()
    lines = []
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as zf:
        offset = 0
        while True:
            batch = _query("SELECT * FROM clips ORDER BY at, id LIMIT 500 OFFSET ?", (offset,))
            if not batch:
                break
            offset += len(batch)
            for row in batch:
                path = wav_path(row["id"])
                if not path or not path.is_file():
                    continue
                name = f"{folder(row)}/{row['id']}.wav"
                zf.write(path, name)
                lines.append(json.dumps(dict(row, file=name), ensure_ascii=False))
                yield sink.take()
        zf.writestr("metadata.jsonl", "\n".join(lines) + ("\n" if lines else ""))
    yield sink.take()
