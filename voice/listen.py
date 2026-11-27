"""The kiosk's listen websocket: ``/voice/api/listen?display=<ip>&v=<version>``.

Each screen streams its microphone as binary frames of 16 kHz signed 16-bit
little-endian mono PCM (20 ms = 640 bytes, but any size is buffered). The app
answers with text JSON:

- ``{"type": "hello", "words": [...], "threshold": 0.5, "assistant": true,
  "ping": 15}`` once accepted (and again if the wake word is changed in
  settings); ``ping`` promises a ``{"type": "ping"}`` every that many seconds;
- ``{"type": "ping"}`` every :data:`PING_SECONDS`, so a screen can tell a live
  socket from one whose far end is gone (websocket pings stop at the hub's
  proxy) and reconnect;
- ``{"type": "wake", "word": ..., "score": 0.91, "at": "<iso8601>"}`` per
  detection, at most one per :data:`DEBOUNCE_SECONDS` of audio;
- ``{"type": "config", "listen": false}`` then a close when listening is off.

After a wake the voice assistant (:mod:`voice.assistant`) may hold a
conversation on the same socket: it sends ``state``/``transcript``/
``speak_*``/``conversation_end`` messages and the reply audio as binary
frames (24 kHz int16 mono). Wake detection is paused on a screen while its
conversation runs. The screen may send ``{"type": "end"}`` to end it.

There is no sign-in (the kiosk has no cookies, like the cameras talk socket):
only private (LAN) addresses are accepted, anything else is closed with
:data:`FORBIDDEN`. Each connection has its own detector; inference runs in a
small thread pool so the event loop only shuffles bytes.
"""
import asyncio
import ipaddress
import itertools
import json
import logging
import math
import os
import threading
import time
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import assistant, clips, config, dataset
from .detector import CHUNK, SAMPLE_RATE, Detector

log = logging.getLogger(__name__)

FORBIDDEN = 4403
DEBOUNCE_SECONDS = 2.0
PEAK_WINDOW = 10.0  # seconds of scores behind "peak score" on the status page
MAX_BACKLOG_SECONDS = 5.0  # audio waiting for the detector before the oldest is dropped
LEVEL_FLOOR = -96.0
PING_SECONDS = 15  # app-level keepalive; screens reconnect after ~3 missed

_executor = ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 1)),
                               thread_name_prefix="voice")
_ids = itertools.count(1)


# --- connected screens ------------------------------------------------------------

class Screen:
    """What the status page shows about one connection."""

    def __init__(self, ip, display, version):
        self.id = next(_ids)
        self.ip = ip
        self.display = display or ip
        self.version = version
        self.since = datetime.now(timezone.utc)
        self.level = LEVEL_FLOOR
        self.heard = 0  # samples processed and answered
        self._scores = deque()
        self._lock = threading.Lock()

    def add_score(self, score, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self._scores.append((now, score))
            while self._scores and now - self._scores[0][0] > PEAK_WINDOW:
                self._scores.popleft()

    def peak(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            return max((s for t, s in self._scores if now - t <= PEAK_WINDOW), default=0.0)

    def as_dict(self):
        conv = assistant.active(self)
        return {"id": self.id, "ip": self.ip, "display": self.display, "version": self.version,
                "since": self.since.isoformat(), "level": round(self.level, 1),
                "peak": round(self.peak(), 3),
                "conversation": conv.as_dict() if conv else None}


_screens = {}
_screens_lock = threading.Lock()


def screens():
    with _screens_lock:
        items = list(_screens.values())
    return [s.as_dict() for s in sorted(items, key=lambda s: s.since)]


def _register(screen):
    with _screens_lock:
        _screens[screen.id] = screen


def _unregister(screen):
    with _screens_lock:
        _screens.pop(screen.id, None)


# --- per-connection audio processing --------------------------------------------

def level_dbfs(samples):
    if not len(samples):
        return LEVEL_FLOOR
    rms = math.sqrt(float(np.mean(np.square(samples, dtype=np.float64))))
    return max(LEVEL_FLOOR, 20 * math.log10(rms / 32768)) if rms > 0 else LEVEL_FLOOR


class Session:
    """Detector, clip ring and debounce for one connection. Runs in the thread pool."""

    def __init__(self, screen, cfg):
        self.screen = screen
        self.recorder = clips.ClipRecorder()
        self._odd = b""
        self.samples = 0  # audio received, for debouncing in audio time
        self._last_wake = None
        self.tail = None  # audio after the last wake in the latest batch
        self.events = dataset.Events(DEBOUNCE_SECONDS)
        self.wake_clip = None  # training clip id of the last wake in the latest batch
        self.reset(cfg)

    def reset(self, cfg):
        self.cfg = cfg
        self.detector = Detector(cfg.word)

    def _pcm(self, data):
        data = self._odd + data
        cut = len(data) - len(data) % 2
        self._odd = data[cut:]
        pcm = np.frombuffer(data[:cut], dtype="<i2").astype(np.int16)
        if len(pcm):
            self.screen.level = level_dbfs(pcm[-SAMPLE_RATE // 2:])
            self.samples += len(pcm)
        return pcm

    def _clips(self, pcm):
        for tag, audio in self.recorder.add(pcm):
            if isinstance(tag, dict):
                dataset.writer.clip(tag, audio)  # a training clip: saved off this thread
                continue
            try:
                clips.save_clip(tag, audio)
            except OSError:
                log.exception("could not save clip %s", tag)

    def _collect(self, score, at_sample, fired, wake_threshold):
        """A detection event peaked at ``at_sample``: ask for its training clip."""
        clip_id = dataset.new_id()
        self.recorder.mark({
            "id": clip_id, "at": datetime.now(timezone.utc).isoformat(), "ip": self.screen.ip,
            "screen": self.screen.display, "word": self.cfg.word, "score": round(score, 3),
            "threshold": wake_threshold, "collect_threshold": self.cfg.collect_threshold,
            "fired": int(fired),
        }, at_sample)
        if fired:
            self.wake_clip = clip_id

    def take(self, data):
        """Received bytes while a conversation runs: no detection; return the PCM."""
        pcm = self._pcm(data)
        if len(pcm):
            self._clips(pcm)
        return pcm

    def process(self, data):
        """Handle received bytes; return the wake messages to send.

        After a wake, :attr:`tail` holds the audio that followed it in this batch.
        """
        self.tail = None
        self.wake_clip = None
        start = self.samples
        pcm = self._pcm(data)
        if not len(pcm):
            return []
        wakes = []
        # Each score belongs to the chunk that ended at that point in the stream.
        pending = self.detector.buffered
        for i, score in enumerate(self.detector.feed(pcm)):
            self.screen.add_score(score)
            at_sample = start - pending + (i + 1) * CHUNK
            fired = score >= self.cfg.threshold and (
                self._last_wake is None
                or at_sample - self._last_wake >= DEBOUNCE_SECONDS * SAMPLE_RATE)
            if self.cfg.collect_clips:
                event = self.events.score(score, at_sample,
                                          min(self.cfg.collect_threshold, self.cfg.threshold), fired)
                if event:
                    self._collect(*event, self.cfg.threshold)
            if not fired:
                continue
            self._last_wake = at_sample
            wakes.append(self._wake(score, at_sample))
            # The rest of the batch belongs to the conversation this wake may start.
            self.tail = pcm[max(0, at_sample - start):]
            break
        self._clips(pcm)
        return wakes

    def _wake(self, score, at_sample):
        now = datetime.now(timezone.utc)
        word = self.cfg.word
        entry = {
            "id": f"{now.strftime('%Y%m%dT%H%M%S%f')}-{self.screen.id}",
            "at": now.isoformat(), "screen": self.screen.display, "ip": self.screen.ip,
            "word": word, "score": round(score, 3), "clip": "",
        }
        # Debug clips are guest speech too: never kept while guests are staying.
        if self.cfg.save_clips and not dataset.writer.stays.guest_staying():
            entry["clip"] = entry["id"] + ".wav"
            self.recorder.mark(entry["clip"], at_sample)
        log.info("wake %s %.2f from %s (%s)", word, score, self.screen.display, self.screen.ip)
        try:
            clips.record(entry)
        except OSError:
            log.exception("could not log detection")
        return {"type": "wake", "word": word, "score": round(score, 3), "at": entry["at"]}


# --- the websocket ---------------------------------------------------------------

def _query(scope):
    q = urllib.parse.parse_qs((scope.get("query_string") or b"").decode("latin1"))
    return {k: (v[0] if v else "") for k, v in q.items()}


def _header(scope, name):
    for key, value in scope.get("headers") or ():
        if key.decode("latin1").lower() == name:
            return value.decode("latin1")
    return ""


def is_private(ip):
    try:
        addr = ipaddress.ip_address((ip or "").strip().strip("[]"))
    except ValueError:
        return False
    if getattr(addr, "ipv4_mapped", None):
        addr = addr.ipv4_mapped
    return addr.is_private or addr.is_loopback or addr.is_link_local


def client_ip(scope):
    """The screen's address, or "" if it can't be trusted.

    In the container every connection comes from the hub's proxy, which
    replaces ``X-Forwarded-For`` with the visitor's address; it is only
    believed when the peer itself is on the LAN or loopback.
    """
    peer = (scope.get("client") or ("", 0))[0] or ""
    if not is_private(peer):
        return ""
    forwarded = _header(scope, "x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return peer


async def _text(send, payload):
    await send({"type": "websocket.send", "text": json.dumps(payload)})


async def _config(refresh=False):
    if refresh:
        config.forget()
    return await asyncio.get_running_loop().run_in_executor(_executor, config.current)


def hello(cfg):
    return {"type": "hello", "words": [cfg.word], "threshold": cfg.threshold,
            "assistant": cfg.assistant, "ping": PING_SECONDS}


async def listen(scope, receive, send):
    message = await receive()
    if message.get("type") != "websocket.connect":
        return
    ip = client_ip(scope)
    if not is_private(ip):
        log.warning("refused listen socket from %s", ip or (scope.get("client") or ("?",))[0])
        await send({"type": "websocket.close", "code": FORBIDDEN})
        return
    query = _query(scope)
    cfg = await _config()
    await send({"type": "websocket.accept"})
    if not cfg.listen:
        await _text(send, {"type": "config", "listen": False})
        await send({"type": "websocket.close", "code": 1000})
        return
    screen = Screen(ip, query.get("display", "")[:64], query.get("v", "")[:32])
    loop = asyncio.get_running_loop()
    session = await loop.run_in_executor(_executor, Session, screen, cfg)
    out = assistant.Out(send)
    await out.text(hello(cfg))
    _register(screen)
    log.info("screen %s connected from %s (kiosk %s)", screen.display, ip, screen.version or "?")
    inbox = asyncio.Queue()
    worker = asyncio.ensure_future(_work(session, inbox, out))
    pinger = asyncio.ensure_future(_keepalive(out))
    try:
        while True:
            message = await receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                inbox.put_nowait(message["bytes"])
            elif message.get("text"):
                await _control(screen, message["text"])
            if worker.done():
                break
    finally:
        out.closed = True
        pinger.cancel()
        await assistant.end(screen, "disconnected")
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("screen %s: listen worker failed", screen.display)
        _unregister(screen)
        log.info("screen %s disconnected", screen.display)


async def _keepalive(out):
    while not out.closed:
        await asyncio.sleep(PING_SECONDS)
        await out.text({"type": "ping"})


async def _control(screen, text):
    try:
        msg = json.loads(text)
    except ValueError:
        return
    if isinstance(msg, dict) and msg.get("type") == "end":
        await assistant.end(screen, "ended")


async def _work(session, inbox, out):
    loop = asyncio.get_running_loop()
    max_backlog = int(MAX_BACKLOG_SECONDS * SAMPLE_RATE * 2)
    checked = time.monotonic()
    talking = False
    while True:
        parts = [await inbox.get()]
        while not inbox.empty():
            parts.append(inbox.get_nowait())
        data = b"".join(parts)
        if len(data) > max_backlog:
            log.warning("screen %s: detector fell behind, dropped %.1f s of audio",
                        session.screen.display, (len(data) - max_backlog) / (SAMPLE_RATE * 2))
            data = data[-max_backlog:]
            data = data[len(data) % 2:]
        if time.monotonic() - checked > config.TTL:
            checked = time.monotonic()
            cfg = await _config()
            if not cfg.listen:
                await assistant.end(session.screen, "disabled")
                await out.text({"type": "config", "listen": False})
                await out.close()
                return
            if cfg.word != session.cfg.word or cfg.assistant != session.cfg.assistant:
                if cfg.word != session.cfg.word:
                    await loop.run_in_executor(_executor, session.reset, cfg)
                await out.text(hello(cfg))
            session.cfg = cfg
        conv = assistant.active(session.screen)
        if conv:
            talking = True
            conv.feed(await loop.run_in_executor(_executor, session.take, data))
        else:
            if talking:
                # Start detection afresh: the old window still holds the wake word.
                talking = False
                await loop.run_in_executor(_executor, session.reset, session.cfg)
            wakes = await loop.run_in_executor(_executor, session.process, data)
            for wake in wakes:
                await out.text(wake)
            if wakes and await assistant.on_wake(session.screen, out, session.cfg, session.tail,
                                                 clip_id=session.wake_clip):
                talking = True
        session.screen.heard = session.samples
