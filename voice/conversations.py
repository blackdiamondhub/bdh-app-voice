"""The durable record of every assistant conversation.

Each conversation gets a ``conversation_id`` and its turns numbered ``seq``
1, 2, 3… in order. One background thread (:class:`Recorder`) writes the
record to a small SQLite database in the app's data directory
(``/data/voice/conversations.sqlite3`` in the container) and only then
publishes it on the hub bus (``voice/conversation``), so nothing is published
that wasn't stored first:

- a ``turn`` event per finished turn: the turn plus ``conversation_id`` and ``seq``;
- one ``final`` event per conversation with the whole record: screen, mode,
  stay, start and end, end reason, the turns with their tool calls, and how
  complete the record is.

``completeness`` is ``complete`` when everything said and done was stored,
``partial`` when something is known to be missing (a write failed, a text was
cut) and ``unknown`` when the app stopped mid-conversation and the record was
recovered on the next start. A consumer (the sentinel app) must never treat
missing data as evidence.

Final events are live only, so a consumer that was down can ask for them
again: it publishes ``sentinel/sync`` with ``{"since": <ISO time>}`` and the
recorder re-publishes the final records that ended since then (``replay``).

Transcripts are kept for admins (Settings → Conversations) only as long as
the retention settings allow: conversations during a guest stay until
checkout plus ``transcript_guest_days``, all others ``transcript_days`` after
they started. ``end_conversation`` is a tool call like any other in the counts.
"""
import json
import logging
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone

from bdh_sdk import events

from . import clips

log = logging.getLogger(__name__)

TOPIC = "voice/conversation"
SYNC_TOPIC = "sentinel/sync"

COMPLETE = "complete"
PARTIAL = "partial"
UNKNOWN = "unknown"

GUEST = "guest"  # a guest is staying
OWNER = "owner"  # the owners are staying
NONE = "none"  # between stays
STAY_UNKNOWN = "unknown"  # the hub couldn't say: treated like a guest stay

DEFAULT_GUEST_DAYS = 7
DEFAULT_DAYS = 30
MAX_TEXT = 4000  # characters per side of a turn
MAX_TURNS = 200
RESULT_CHARS = 200
MAX_EVENT_BYTES = 200_000  # the broker takes 256 KB per call
TRIMMED_TEXT = 500
MAX_REPLAY = 100
MAX_REPLAY_DAYS = 7
PRUNE_EVERY = 3600.0  # seconds
QUEUE_SIZE = 256

_CONV_COLUMNS = (
    ("id", "TEXT PRIMARY KEY"), ("sid", "TEXT"), ("screen", "TEXT"), ("ip", "TEXT"),
    ("mode", "TEXT"), ("model", "TEXT"), ("stay", "TEXT"), ("stay_key", "TEXT"),
    ("stay_checkout", "TEXT"), ("started_at", "TEXT"), ("ended_at", "TEXT"),
    ("end_reason", "TEXT"), ("seconds", "REAL"), ("turn_count", "INTEGER"),
    ("user_turns", "INTEGER"), ("replies", "INTEGER"), ("tool_calls", "INTEGER"),
    ("completeness", "TEXT"), ("missing", "TEXT"), ("final", "INTEGER"),
)
CONV_FIELDS = tuple(name for name, _ in _CONV_COLUMNS)

_db_lock = threading.RLock()


def private(stay):
    """Whether a conversation in ``stay`` (a stay kind) must keep its words out of the log."""
    return stay not in (OWNER, NONE)


def now_iso(now=None):
    return (now or datetime.now(timezone.utc)).isoformat()


def new_id():
    return uuid.uuid4().hex


def _parse(iso):
    try:
        when = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


# --- the stay ----------------------------------------------------------------------

def stay_info(stays=None):
    """``{"stay", "stay_key", "stay_checkout"}`` for a conversation starting now.

    Asks :class:`voice.dataset.StayCheck` (shared with the clip collector). When
    the hub can't say, the stay is ``unknown``, which is treated as a guest stay.
    """
    if stays is None:
        from . import dataset

        stays = dataset.writer.stays
    info = getattr(stays, "info", None)
    if callable(info):
        return info()
    return {"stay": GUEST if stays.guest_staying() else NONE, "stay_key": "", "stay_checkout": ""}


# --- the database ------------------------------------------------------------------

def path():
    return clips.data_dir() / "conversations.sqlite3"


def _connect():
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(p), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS conversations (%s)"
               % ", ".join(f"{n} {t}" for n, t in _CONV_COLUMNS))
    db.execute("CREATE INDEX IF NOT EXISTS conversations_started ON conversations (started_at)")
    db.execute("CREATE INDEX IF NOT EXISTS conversations_ended ON conversations (ended_at)")
    db.execute("CREATE TABLE IF NOT EXISTS turns (conversation_id TEXT, seq INTEGER, at TEXT, "
               "user TEXT, assistant TEXT, tools TEXT, PRIMARY KEY (conversation_id, seq))")
    return db


def _query(sql, args=()):
    with _db_lock:
        db = _connect()
        try:
            with db:
                return [dict(r) for r in db.execute(sql, args)]
        finally:
            db.close()


def _execute(sql, args=()):
    with _db_lock:
        db = _connect()
        try:
            with db:
                return db.execute(sql, args).rowcount
        finally:
            db.close()


def _upsert(row):
    """Insert or update a conversation row with the given fields."""
    fields = [k for k in CONV_FIELDS if k in row]
    with _db_lock:
        db = _connect()
        try:
            with db:
                db.execute("INSERT INTO conversations (%s) VALUES (%s) ON CONFLICT(id) DO UPDATE SET %s"
                           % (", ".join(fields), ", ".join("?" * len(fields)),
                              ", ".join(f"{k} = excluded.{k}" for k in fields if k != "id")),
                           [row[k] for k in fields])
        finally:
            db.close()


def _insert_turn(conversation_id, turn):
    with _db_lock:
        db = _connect()
        try:
            with db:
                db.execute("INSERT OR REPLACE INTO turns (conversation_id, seq, at, user, assistant, tools) "
                           "VALUES (?, ?, ?, ?, ?, ?)",
                           (conversation_id, turn["seq"], turn["at"], turn["user"], turn["assistant"],
                            json.dumps(turn["tools"], default=str)))
        finally:
            db.close()


def _turns(conversation_id):
    out = []
    for r in _query("SELECT seq, at, user, assistant, tools FROM turns WHERE conversation_id = ? "
                    "ORDER BY seq", (conversation_id,)):
        try:
            r["tools"] = json.loads(r["tools"] or "[]")
        except ValueError:
            r["tools"] = []
        out.append(r)
    return out


def _row(r):
    r = dict(r)
    r["missing"] = [m for m in (r.get("missing") or "").split(",") if m]
    r["final"] = bool(r.get("final"))
    return r


def get(conversation_id):
    """One conversation with its ``turns``, or None."""
    found = _query("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
    if not found:
        return None
    rec = _row(found[0])
    rec["turns"] = _turns(conversation_id)
    return rec


def rows(limit=50, offset=0):
    """Conversations without their turns, newest first."""
    return [_row(r) for r in _query("SELECT * FROM conversations ORDER BY started_at DESC, id DESC "
                                    "LIMIT ? OFFSET ?", (int(limit), int(offset)))]


def count():
    return _query("SELECT COUNT(*) AS n FROM conversations")[0]["n"]


def _delete(ids):
    if not ids:
        return 0
    with _db_lock:
        db = _connect()
        try:
            with db:
                db.executemany("DELETE FROM turns WHERE conversation_id = ?", [(i,) for i in ids])
                db.executemany("DELETE FROM conversations WHERE id = ?", [(i,) for i in ids])
        finally:
            db.close()
    return len(ids)


# --- retention ----------------------------------------------------------------------

def expires_at(row, guest_days=DEFAULT_GUEST_DAYS, days=DEFAULT_DAYS):
    """When a conversation's record is deleted.

    During a guest stay: checkout + ``guest_days``. Everything else (owner
    stays, between stays, or a guest stay whose checkout isn't known):
    ``days`` after it started.
    """
    started = _parse(row.get("started_at")) or datetime.now(timezone.utc)
    if row.get("stay") == GUEST:
        checkout = _parse(row.get("stay_checkout"))
        if checkout:
            return checkout + timedelta(days=guest_days)
    return started + timedelta(days=days)


def prune(now=None, guest_days=DEFAULT_GUEST_DAYS, days=DEFAULT_DAYS):
    """Delete the records whose time is up; returns how many."""
    now = now or datetime.now(timezone.utc)
    doomed = [r["id"] for r in _query("SELECT id, stay, stay_checkout, started_at FROM conversations")
              if expires_at(r, guest_days, days) <= now]
    return _delete(doomed)


def recover(open_ids=(), now=None):
    """Close the records of conversations that never finished (the app stopped).

    They become ``unknown``: some of what was said may be missing. Returns their ids.
    """
    lost = [r["id"] for r in _query("SELECT id FROM conversations WHERE final = 0")
            if r["id"] not in open_ids]
    for conversation_id in lost:
        rec = get(conversation_id)
        last = rec["turns"][-1]["at"] if rec["turns"] else None
        _upsert({"id": conversation_id, "ended_at": last or rec.get("started_at") or now_iso(now),
                 "end_reason": "lost", "completeness": UNKNOWN,
                 "missing": ",".join(sorted(set(rec["missing"]) | {"app_restarted"})),
                 "turn_count": len(rec["turns"]), "final": 1})
    return lost


# --- events ---------------------------------------------------------------------------

def final_event(rec, replay=False):
    """The ``final`` event for a stored record (trimmed to fit the broker)."""
    data = {
        "kind": "final", "conversation_id": rec["id"], "sid": rec.get("sid") or "",
        "screen": rec.get("screen") or "", "ip": rec.get("ip") or "", "mode": rec.get("mode") or "",
        "model": rec.get("model") or "", "stay": rec.get("stay") or "",
        "stay_key": rec.get("stay_key") or "", "started_at": rec.get("started_at"),
        "ended_at": rec.get("ended_at"), "end_reason": rec.get("end_reason") or "",
        "seconds": rec.get("seconds"), "turn_count": rec.get("turn_count") or 0,
        "user_turns": rec.get("user_turns") or 0, "replies": rec.get("replies") or 0,
        "tool_calls": rec.get("tool_calls") or 0, "completeness": rec.get("completeness") or UNKNOWN,
        "missing": list(rec.get("missing") or ()), "turns": rec.get("turns") or [], "replay": replay,
    }
    if len(json.dumps(data, default=str)) > MAX_EVENT_BYTES:
        data["turns"] = [dict(t, user=(t.get("user") or "")[:TRIMMED_TEXT],
                              assistant=(t.get("assistant") or "")[:TRIMMED_TEXT],
                              tools=[{k: v for k, v in c.items() if k != "result"}
                                     for c in t.get("tools") or ()])
                         for t in data["turns"]]
        if data["completeness"] == COMPLETE:
            data["completeness"] = PARTIAL
        data["missing"] = sorted(set(data["missing"]) | {"event_trimmed"})
    return data


def _publish(data):
    try:
        events.publish(TOPIC, data)
    except Exception:
        log.exception("could not publish the conversation record")


# --- the recorder thread -------------------------------------------------------------

class Recorder:
    """Stores conversation records in order, on its own thread, then publishes them."""

    def __init__(self, settings=None):
        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread = None
        self._lock = threading.Lock()
        self._settings = settings
        self._open = set()  # conversations started in this process and not finished
        self._failed = {}  # conversation id -> reasons something is missing
        self._pruned_at = None
        self.dropped = 0

    def _cfg(self):
        if self._settings is not None:
            return self._settings()
        from . import config

        return config.current()

    def _start(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="voice-conversations",
                                                daemon=True)
                self._thread.start()

    def _put(self, item):
        self._start()
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            log.warning("conversation recorder is behind: dropped a %s (%d so far)", item[0], self.dropped)
            if item[0] in ("turn", "finish"):
                self._failed.setdefault(item[1], set()).add("dropped")
            return False

    # Called from the conversation (never blocks).
    def started(self, conversation_id, meta):
        self._open.add(conversation_id)
        return self._put(("start", conversation_id, meta))

    def turn(self, conversation_id, turn, header):
        return self._put(("turn", conversation_id, (turn, header)))

    def finished(self, conversation_id, summary):
        return self._put(("finish", conversation_id, summary))

    def replay(self, since):
        return self._put(("replay", since, None))

    def flush(self, timeout=10):
        """Wait until everything queued so far is stored and published (for tests)."""
        done = threading.Event()
        self._start()
        self._queue.put(("flush", done, None), timeout=timeout)
        return done.wait(timeout)

    def _run(self):
        self._housekeeping(force=True)
        while True:
            try:
                kind, a, b = self._queue.get(timeout=PRUNE_EVERY)
            except queue.Empty:
                self._housekeeping()
                continue
            try:
                if kind == "flush":
                    a.set()
                elif kind == "start":
                    self._store_start(a, b)
                elif kind == "turn":
                    self._store_turn(a, *b)
                elif kind == "finish":
                    self._store_finish(a, b)
                    self._housekeeping()
                elif kind == "replay":
                    self._replay(a)
            except Exception:
                log.exception("conversation recorder: could not handle %s", kind)
                if kind in ("start", "turn", "finish"):
                    self._failed.setdefault(a, set()).add("write_failed")

    def _housekeeping(self, force=False):
        now = datetime.now(timezone.utc)
        if not force and self._pruned_at and (now - self._pruned_at).total_seconds() < PRUNE_EVERY:
            return
        self._pruned_at = now
        try:
            for conversation_id in recover(set(self._open), now):
                rec = get(conversation_id)
                if rec:
                    _publish(final_event(rec))
            cfg = self._cfg()
            prune(now, cfg.transcript_guest_days, cfg.transcript_days)
        except Exception:
            log.exception("conversation recorder: housekeeping failed")

    def _store_start(self, conversation_id, meta):
        _upsert(dict(meta, id=conversation_id, completeness=PARTIAL, final=0, missing="",
                     turn_count=0))

    def _store_turn(self, conversation_id, turn, header):
        try:
            _insert_turn(conversation_id, turn)
        except Exception:
            self._failed.setdefault(conversation_id, set()).add("write_failed")
            log.exception("could not store a conversation turn")
        _publish({**header, "kind": "turn", "conversation_id": conversation_id, "seq": turn["seq"],
                  "turn": turn["seq"], "at": turn["at"], "user": turn["user"],
                  "assistant": turn["assistant"], "tools": turn["tools"]})

    def _store_finish(self, conversation_id, summary):
        self._open.discard(conversation_id)
        missing = set(summary.pop("missing", ())) | self._failed.pop(conversation_id, set())
        stored = _query("SELECT seq FROM turns WHERE conversation_id = ?", (conversation_id,))
        if len(stored) != summary.get("turn_count", 0):
            missing.add("turns_missing")
        row = dict(summary, id=conversation_id, final=1, missing=",".join(sorted(missing)),
                   completeness=PARTIAL if missing else COMPLETE)
        _upsert(row)
        rec = get(conversation_id)
        if rec:
            _publish(final_event(rec))

    def _replay(self, since):
        start = _parse(since)
        floor = datetime.now(timezone.utc) - timedelta(days=MAX_REPLAY_DAYS)
        if start is None or start < floor:
            start = floor
        found = _query("SELECT id FROM conversations WHERE final = 1 AND ended_at >= ? "
                       "ORDER BY ended_at LIMIT ?", (start.isoformat(), MAX_REPLAY))
        for r in found:
            rec = get(r["id"])
            if rec:
                _publish(final_event(rec, replay=True))
        log.info("replayed %d conversation records since %s", len(found), start.isoformat())


recorder = Recorder()


def on_sync(event):
    """``sentinel/sync``: re-publish the final records that ended since ``since``."""
    data = event.data if isinstance(event.data, dict) else {}
    recorder.replay(str(data.get("since") or ""))


_subscription = None


def subscribe():
    """Answer the sentinel's ``sentinel/sync`` requests (needs the grant; harmless without)."""
    global _subscription
    if _subscription is None:
        try:
            _subscription = events.subscribe(SYNC_TOPIC, on_sync)
        except Exception:
            log.exception("could not subscribe to %s", SYNC_TOPIC)


def start():
    """In the app's container: recover and prune now, and answer sync requests."""
    recorder._start()
    subscribe()


def clip_text(text, limit=MAX_TEXT):
    """``(text, cut)``: ``text`` at most ``limit`` characters."""
    text = text or ""
    return (text[:limit], True) if len(text) > limit else (text, False)
