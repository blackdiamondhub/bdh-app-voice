"""The voice assistant: a speech-to-speech conversation after a wake word.

When a screen's listen socket hears the wake word, :func:`on_wake` starts a
:class:`Conversation` for that screen. It opens an assistant session on the
hub's tools gateway (``bdh_sdk.tools``), connects to the Azure OpenAI
Realtime API over a websocket, streams the screen's microphone to it
(resampled 16 → 24 kHz), runs the model's function calls through the
gateway and streams the spoken reply (24 kHz PCM) back to the screen as
binary frames on the same listen socket, framed by JSON messages:

- ``{"type": "state", "state": "listening"|"thinking"|"speaking"|"idle"}``
- ``{"type": "transcript", "role": "user"|"assistant", "text": ..., "final": bool}``
- ``{"type": "speak_start"}`` … binary audio … ``{"type": "speak_end"}``
- ``{"type": "speak_stop"}``: the user interrupted; drop queued audio now
- ``{"type": "conversation_end", "reason": ..., "message"?: ...}``

The server's semantic VAD ends the user's turns and starts replies. The
conversation ends after ``follow_up_seconds`` without speech once the last
reply has played, once the goodbye has played after the model calls the
built-in :data:`END_TOOL`, after ``max_seconds``, on a realtime error, or when the
screen goes away. Each finished turn is published as a hub event on
``voice/conversation`` for a future supervisor.
"""
import asyncio
import base64
import json
import logging
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import websockets
from bdh_sdk import events, rpc, tools

from . import config
from .audio import MODEL_RATE, Resampler

log = logging.getLogger(__name__)

MAX_CONCURRENT = 2  # realtime conversations at once, across all screens
CONNECT_TIMEOUT = 10.0
MAX_PENDING_SECONDS = 5.0  # mic audio kept while the realtime socket connects
AUDIO_FRAME_BYTES = 9600  # 200 ms of 24 kHz int16 per binary frame to the kiosk
PLAYBACK_MARGIN = 0.5  # seconds of kiosk buffering on top of the audio sent
MAX_TOOL_OUTPUT = 8000  # characters of a tool result given back to the model
EVENT_TOPIC = "voice/conversation"
TRANSCRIPTION_MODEL = "whisper-1"  # the one deployed next to the realtime models
LODGE_UTC_OFFSET = timedelta(hours=-7)  # America/Vancouver: permanent UTC-7 since Nov 2026
# Errors the API sends for races that don't matter (cancel after the reply ended, …).
HARMLESS_ERRORS = frozenset({"response_cancel_not_active", "conversation_already_has_active_response",
                             "item_truncate_invalid_item_id"})
END_TOOL = "end_conversation"  # handled here, not by the gateway
END_TOOL_DEF = {"type": "function", "name": END_TOOL,
                "description": "Stop listening and go back to waiting for the wake word. Call it "
                               "when the person is done, after a short goodbye.",
                "parameters": {"type": "object", "properties": {}}}
NOT_CONFIGURED = "The assistant isn't set up yet."
BUSY = "The assistant is busy, try again in a moment."

INSTRUCTIONS = """\
You are the voice assistant on a wall screen at Black Diamond Lodge, a ski-in ski-out \
vacation home in Sun Peaks, British Columbia. You talk with guests and the family who stay here.
- Be friendly and concise: answer in one or two short spoken sentences. No lists or markdown.
- Use your tools to look things up and to act. Only say something was done when the tool \
answered ok; if it failed, say so briefly.
- If you can't do something here (no suitable tool, or the tool was refused or unavailable), \
say you can't do that from this screen. Never attempt owner or admin actions.
- Never mention the names of your tools or functions, or how you work inside.
- Before using a tool marked "confirm first", ask the guest to confirm and wait for a yes.
- People describe things loosely ("the main fireplace", "the one in here"): match their words \
to the names your tools give. If it could mean more than one thing and the conversation doesn't \
settle it, ask a short question ("The living room or the master bedroom?") before acting; if \
it's clear, just do it without asking.
- When they're done ("thanks", "that's all", "never mind", "stop", "goodbye"), say a very short \
goodbye and call end_conversation in the same reply. Also call it after a goodbye when they've \
clearly finished; otherwise just stop talking and wait, they may follow up.
- If you didn't catch what was said, ask them to repeat it. Reply in the language you're spoken to.
- The current local time at the lodge (Pacific time, UTC-7) is {now}."""

_active = {}  # screen id -> Conversation


def lodge_now(now=None):
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone(LODGE_UTC_OFFSET, "PT"))


def _say_time(t):
    hour = t.hour % 12 or 12
    return f"{t:%A}, {t:%B} {t.day}, {t.year}, {hour}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"


def instructions(cfg, now=None, room=""):
    text = INSTRUCTIONS.format(now=_say_time(lodge_now(now)))
    if room:
        text += (f"\n- This screen is in {room}. \"Here\" or \"this room\" means there; prefer "
                 "what's in this room when a request doesn't say which.")
    if cfg.instructions:
        text += "\n" + cfg.instructions
    return text


def realtime_url(endpoint, model):
    """``wss://<host>/openai/v1/realtime?model=<deployment>`` for an endpoint setting."""
    parts = urllib.parse.urlsplit(endpoint if "://" in endpoint else "https://" + endpoint)
    scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
    path = parts.path.rstrip("/")
    if not path.endswith("/realtime"):
        path += "/openai/v1/realtime"
    return urllib.parse.urlunsplit((scheme, parts.netloc, path,
                                    urllib.parse.urlencode({"model": model}), ""))


def function_tools(listed):
    """The gateway's tools as realtime function tools, plus :data:`END_TOOL`."""
    out = [END_TOOL_DEF]
    for t in listed:
        if t.get("name") == END_TOOL:
            continue
        desc = t.get("description") or ""
        if t.get("title"):
            desc = f"{t['title']}: {desc}" if desc else t["title"]
        if t.get("confirm"):
            desc += " (confirm first)"
        params = t.get("parameters") or {"type": "object", "properties": {}}
        out.append({"type": "function", "name": t["name"], "description": desc,
                    "parameters": params})
    return out


def session_update(cfg, tool_defs, now=None, room=""):
    return {"type": "session.update", "session": {
        "type": "realtime",
        "instructions": instructions(cfg, now, room),
        "output_modalities": ["audio"],
        "tools": tool_defs,
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": MODEL_RATE},
                "noise_reduction": {"type": "far_field"},
                "transcription": {"model": TRANSCRIPTION_MODEL},
                "turn_detection": {"type": "semantic_vad", "eagerness": "auto",
                                   "create_response": True, "interrupt_response": True},
            },
            "output": {"format": {"type": "audio/pcm", "rate": MODEL_RATE}, "voice": cfg.voice},
        },
    }}


class RealtimeError(Exception):
    pass


class Out:
    """Sends to one screen's listen socket; text and binary frames never interleave."""

    def __init__(self, send):
        self._send = send
        self._lock = asyncio.Lock()
        self.closed = False

    async def text(self, payload):
        await self._raw({"type": "websocket.send", "text": json.dumps(payload)})

    async def audio(self, data):
        for i in range(0, len(data), AUDIO_FRAME_BYTES):
            await self._raw({"type": "websocket.send", "bytes": data[i:i + AUDIO_FRAME_BYTES]})

    async def close(self, code=1000):
        await self._raw({"type": "websocket.close", "code": code})

    async def _raw(self, message):
        if self.closed:
            return
        async with self._lock:
            try:
                await self._send(message)
            except Exception:  # the screen went away; the listen loop will notice
                self.closed = True


# --- starting and stopping -------------------------------------------------------

def active(screen):
    return _active.get(screen.id)


def conversations():
    return list(_active.values())


async def on_wake(screen, out, cfg, tail=None):
    """The wake word was heard on ``screen``: start a conversation if we can."""
    if not cfg.assistant or screen.id in _active:
        return None
    key = await asyncio.to_thread(config.api_key)
    if not key:
        log.info("wake on %s but the assistant has no API key", screen.display)
        await out.text({"type": "conversation_end", "reason": "not_configured",
                        "message": NOT_CONFIGURED})
        return None
    if len(_active) >= MAX_CONCURRENT:
        log.info("wake on %s but %d conversations are running", screen.display, len(_active))
        await out.text({"type": "conversation_end", "reason": "busy", "message": BUSY})
        return None
    conv = Conversation(screen, out, cfg, key)
    _active[screen.id] = conv
    if tail is not None and len(tail):
        conv.feed(tail)
    conv.task = asyncio.ensure_future(conv.run())
    return conv


async def end(screen, reason):
    """Stop ``screen``'s conversation (if any) and wait for it to wind down."""
    conv = _active.get(screen.id)
    if conv:
        await conv.stop(reason)


# --- one conversation --------------------------------------------------------------

class Conversation:
    def __init__(self, screen, out, cfg, key):
        self.screen = screen
        self.out = out
        self.cfg = cfg
        self._key = key
        self.task = None
        self.state = ""
        self.reason = ""
        self.started = time.monotonic()
        self.sid = ""
        self.mode = ""
        self.token = None
        self.tools = []
        self.ws = None
        self._resample = Resampler()
        self._pending = []  # resampled audio before the realtime session is ready
        self._pending_bytes = 0
        self._audio = asyncio.Queue()
        self.sent_samples = 0  # 24 kHz samples sent to the model
        # Turn state
        self.user_speaking = False
        self.response_id = None  # response being generated
        self.cancelled = set()
        self.tool_tasks = {}  # call_id -> task
        self.followups = 0  # response.create calls waiting on tools
        self.ending = False  # the model called END_TOOL; end once its goodbye has played
        self.last_activity = time.monotonic()
        # Playback (what the screen is playing)
        self.speaking = False
        self.speak_item = None
        self.speak_started = 0.0
        self.speak_ms = 0.0
        self.playback_end = 0.0
        # The turn being recorded for the event
        self.turn = self._new_turn()
        self.turns = 0
        self._partial = {}  # item id -> text so far

    def __repr__(self):
        return f"<Conversation {self.screen.display} {self.state or 'starting'}>"

    def as_dict(self):
        return {"screen": self.screen.display, "ip": self.screen.ip, "sid": self.sid,
                "state": self.state or "starting", "model": self.cfg.model,
                "seconds": round(time.monotonic() - self.started), "turns": self.turns}

    # --- audio in -----------------------------------------------------------------

    def feed(self, pcm16k):
        """Mic audio from the screen (int16 at 16 kHz). Never blocks."""
        if self.reason:
            return
        out = self._resample(pcm16k)
        if not len(out):
            return
        data = out.tobytes()
        if self.ws is None:
            self._pending.append(data)
            self._pending_bytes += len(data)
            limit = int(MAX_PENDING_SECONDS * MODEL_RATE * 2)
            while self._pending_bytes > limit and len(self._pending) > 1:
                self._pending_bytes -= len(self._pending.pop(0))
            return
        self._audio.put_nowait(data)

    async def _pump_audio(self, ws):
        if self._pending:
            data, self._pending, self._pending_bytes = b"".join(self._pending), [], 0
            await self._append(ws, data)
        while True:
            parts = [await self._audio.get()]
            while not self._audio.empty():
                parts.append(self._audio.get_nowait())
            await self._append(ws, b"".join(parts))

    async def _append(self, ws, data):
        self.sent_samples += len(data) // 2
        await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                  "audio": base64.b64encode(data).decode("ascii")}))

    # --- lifecycle --------------------------------------------------------------------

    async def run(self):
        reason = "error"
        message = ""
        limit = asyncio.timeout(self.cfg.max_seconds)
        try:
            await self._set_state("listening")
            async with limit:
                reason = await self._converse()
        except TimeoutError:
            if limit.expired():
                reason = "max_seconds"
            else:
                log.warning("screen %s: realtime connection timed out", self.screen.display)
                message = "Sorry, I can't reach the assistant right now."
        except asyncio.CancelledError:
            reason = self.reason or "stopped"
        except RealtimeError as exc:
            log.warning("screen %s: realtime error: %s", self.screen.display, exc)
            message = "Sorry, I can't talk right now."
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            log.warning("screen %s: realtime connection failed: %s", self.screen.display, exc)
            message = "Sorry, I can't reach the assistant right now."
        except Exception:
            log.exception("screen %s: conversation failed", self.screen.display)
        finally:
            self.reason = self.reason or reason
            await self._finish(message)

    async def stop(self, reason):
        if not self.reason:
            self.reason = reason
        task = self.task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _finish(self, message):
        for t in list(self.tool_tasks.values()):
            t.cancel()
        if self.speaking and self.reason not in ("idle", "done"):
            await self.out.text({"type": "speak_stop"})
        self.speaking = False
        self._publish_turn()
        await self._set_state("idle")
        end = {"type": "conversation_end", "reason": self.reason}
        if message:
            end["message"] = message
        await self.out.text(end)
        if _active.get(self.screen.id) is self:
            del _active[self.screen.id]
        log.info("screen %s: conversation ended (%s) after %.0f s, %d turns", self.screen.display,
                 self.reason, time.monotonic() - self.started, self.turns)

    async def _set_state(self, state):
        if state != self.state:
            self.state = state
            await self.out.text({"type": "state", "state": state})

    async def _open_tools(self):
        """Open a gateway session and list its tools; no tools if refused."""
        def go():
            res = rpc.client().call("tools.open", {"kind": tools.VOICE, "display": self.screen.ip})
            token = res["token"]
            return token, res.get("sid", ""), res.get("mode", ""), tools.list_tools(token)
        try:
            self.token, self.sid, self.mode, self.tools = await asyncio.to_thread(go)
        except Exception as exc:
            log.warning("screen %s: tools gateway unavailable: %s", self.screen.display, exc)
            self.token, self.tools = None, []
        log.info("screen %s: conversation %s (%s) with %d tools", self.screen.display,
                 self.sid or "-", self.mode or "no gateway", len(self.tools))

    async def _converse(self):
        await self._open_tools()
        url = realtime_url(self.cfg.endpoint, self.cfg.model)
        async with websockets.connect(url, additional_headers={"api-key": self._key},
                                      open_timeout=CONNECT_TIMEOUT, max_size=16 * 2 ** 20,
                                      ping_interval=20, ping_timeout=20) as ws:
            await self._start_session(ws)
            self.ws = ws
            self.last_activity = time.monotonic()
            pump = asyncio.ensure_future(self._pump_audio(ws))
            watch = asyncio.ensure_future(self._watch_idle())
            reader = asyncio.ensure_future(self._read(ws))
            try:
                done, _ = await asyncio.wait({pump, watch, reader},
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    exc = t.exception()
                    if exc:
                        raise exc
                if reader in done:
                    raise RealtimeError("the realtime socket closed")
                return "done" if self.ending else "idle"
            finally:
                for t in (pump, watch, reader):
                    t.cancel()
                await asyncio.gather(pump, watch, reader, return_exceptions=True)

    async def _start_session(self, ws):
        async def next_event():
            ev = json.loads(await asyncio.wait_for(ws.recv(), CONNECT_TIMEOUT))
            if ev.get("type") == "error":
                err = ev.get("error") or {}
                raise RealtimeError(f"{err.get('code') or err.get('type')}: {err.get('message')}")
            return ev

        ev = await next_event()
        if ev.get("type") != "session.created":
            raise RealtimeError(f"expected session.created, got {ev.get('type')}")
        room = config.room_for(self.cfg, self.screen.ip, self.screen.display)
        await ws.send(json.dumps(session_update(self.cfg, function_tools(self.tools), room=room)))
        while (await next_event()).get("type") != "session.updated":
            pass

    async def _watch_idle(self):
        while True:
            await asyncio.sleep(0.2)
            now = time.monotonic()
            busy = (self.user_speaking or self.response_id or self.tool_tasks or self.followups)
            if busy:
                continue
            quiet_since = max(self.last_activity, self.playback_end)
            if self.ending and now >= self.playback_end:
                return
            if now - quiet_since >= self.cfg.follow_up_seconds:
                return

    # --- events from the model ---------------------------------------------------------

    async def _read(self, ws):
        async for raw in ws:
            if isinstance(raw, bytes):
                continue
            ev = json.loads(raw)
            handler = getattr(self, "_on_" + ev.get("type", "").replace(".", "_"), None)
            if handler:
                await handler(ws, ev)

    async def _on_error(self, ws, ev):
        err = ev.get("error") or {}
        if err.get("code") in HARMLESS_ERRORS:
            return
        log.warning("screen %s: realtime error event %s: %s", self.screen.display,
                    err.get("code") or err.get("type"), err.get("message"))

    async def _on_input_audio_buffer_speech_started(self, ws, ev):
        self.user_speaking = True
        self.ending = False  # "oh, one more thing…"
        self.last_activity = time.monotonic()
        if self._playing():
            await self._barge_in(ws, ev)
        await self._set_state("listening")

    async def _on_input_audio_buffer_speech_stopped(self, ws, ev):
        self.user_speaking = False
        self.last_activity = time.monotonic()
        await self._set_state("thinking")

    async def _on_conversation_item_input_audio_transcription_delta(self, ws, ev):
        key = ev.get("item_id", "")
        self._partial[key] = self._partial.get(key, "") + (ev.get("delta") or "")
        await self.out.text({"type": "transcript", "role": "user", "text": self._partial[key],
                             "final": False})

    async def _on_conversation_item_input_audio_transcription_completed(self, ws, ev):
        self._partial.pop(ev.get("item_id", ""), None)
        text = (ev.get("transcript") or "").strip()
        if text:
            self.turn["user"] = (self.turn["user"] + " " + text).strip()
            await self.out.text({"type": "transcript", "role": "user", "text": text, "final": True})

    async def _on_response_created(self, ws, ev):
        self.response_id = (ev.get("response") or {}).get("id") or "?"
        self.last_activity = time.monotonic()
        if self.followups == 0 and not self.user_speaking:
            await self._set_state("thinking")

    async def _on_response_output_audio_delta(self, ws, ev):
        if ev.get("response_id") in self.cancelled:
            return
        data = base64.b64decode(ev.get("delta") or "")
        if not data:
            return
        now = time.monotonic()
        if not self.speaking:
            self.speaking = True
            self.speak_item = ev.get("item_id")
            self.speak_started = now
            self.speak_ms = 0.0
            await self.out.text({"type": "speak_start"})
            await self._set_state("speaking")
        self.speak_ms += len(data) / (MODEL_RATE * 2) * 1000
        self.playback_end = self.speak_started + self.speak_ms / 1000 + PLAYBACK_MARGIN
        await self.out.audio(data)

    async def _on_response_output_audio_done(self, ws, ev):
        if self.speaking and ev.get("response_id") not in self.cancelled:
            self.speaking = False
            await self.out.text({"type": "speak_end"})

    async def _on_response_output_audio_transcript_delta(self, ws, ev):
        if ev.get("response_id") in self.cancelled:
            return
        key = ev.get("item_id", "")
        self._partial[key] = self._partial.get(key, "") + (ev.get("delta") or "")
        await self.out.text({"type": "transcript", "role": "assistant", "text": self._partial[key],
                             "final": False})

    async def _on_response_output_audio_transcript_done(self, ws, ev):
        self._partial.pop(ev.get("item_id", ""), None)
        text = (ev.get("transcript") or "").strip()
        if not text:
            return
        if ev.get("response_id") in self.cancelled:
            text += " …"
        self.turn["assistant"] = (self.turn["assistant"] + " " + text).strip()
        await self.out.text({"type": "transcript", "role": "assistant", "text": text, "final": True})

    async def _on_response_output_item_done(self, ws, ev):
        item = ev.get("item") or {}
        if item.get("type") != "function_call" or ev.get("response_id") in self.cancelled:
            return
        call_id = item.get("call_id") or ""
        if item.get("name") == END_TOOL:
            self.ending = True
            self.turn["tools"].append({"name": END_TOOL, "args": {}, "ok": True})
            log.info("screen %s: the assistant ended the conversation", self.screen.display)
            return
        self.tool_tasks[call_id] = asyncio.ensure_future(self._run_tool(ws, item))

    async def _on_response_done(self, ws, ev):
        response = ev.get("response") or {}
        if self.response_id in (response.get("id"), "?") or not response.get("id"):
            self.response_id = None
        self.last_activity = time.monotonic()
        calls = [o.get("call_id") for o in response.get("output") or ()
                 if o.get("type") == "function_call" and o.get("call_id") and o.get("name") != END_TOOL]
        if response.get("status") == "cancelled":
            return  # the user spoke over it; their next turn gets its own reply
        if calls:
            self.followups += 1
            await self._set_state("thinking")
            asyncio.ensure_future(self._continue_after(ws, calls))
            return
        self.turns += 1
        self._publish_turn()
        if not self.user_speaking:
            await self._set_state("listening")

    def _playing(self):
        return self.speaking or time.monotonic() < self.playback_end - PLAYBACK_MARGIN

    async def _barge_in(self, ws, ev):
        """The user spoke over the reply: stop the screen and the model."""
        now = time.monotonic()
        played_ms = int(max(0.0, min(self.speak_ms, (now - self.speak_started) * 1000)))
        await self.out.text({"type": "speak_stop"})
        self.speaking = False
        self.playback_end = now
        if self.speak_item:
            await ws.send(json.dumps({"type": "conversation.item.truncate",
                                      "item_id": self.speak_item, "content_index": 0,
                                      "audio_end_ms": played_ms}))
        if self.response_id:
            self.cancelled.add(self.response_id)
            await ws.send(json.dumps({"type": "response.cancel"}))
        self.speak_item = None
        log.info("screen %s: barge-in after %d ms of reply", self.screen.display, played_ms)

    # --- tools ----------------------------------------------------------------------

    def _call(self, name, args):
        """Run a tool through the gateway (in a thread). Reopens an expired session once."""
        if self.token is None:
            return {"ok": False, "error": "no tools are available on this screen"}
        for attempt in (1, 2):
            try:
                return tools.call_tool(self.token, name, args)
            except rpc.BrokerError as exc:
                if exc.code == tools.BAD_SESSION and attempt == 1:
                    try:
                        self.token = tools.open_session(tools.VOICE, display=self.screen.ip)
                        continue
                    except Exception:
                        pass
                log.warning("screen %s: tool %s refused: %s", self.screen.display, name, exc)
                return {"ok": False, "error": "the hub refused the request"}
            except Exception as exc:
                log.warning("screen %s: tool %s failed: %s", self.screen.display, name, exc)
                return {"ok": False, "error": "the tool is unavailable right now"}
        return {"ok": False, "error": "the tool is unavailable right now"}

    async def _run_tool(self, ws, item):
        name = item.get("name") or ""
        call_id = item.get("call_id") or ""
        try:
            args = json.loads(item.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError
        except ValueError:
            result = {"ok": False, "error": "the arguments were not valid JSON"}
            args = None
        else:
            result = await asyncio.to_thread(self._call, name, args)
        ok = bool(result.get("ok"))
        summary = {"name": name, "args": args, "ok": ok}
        if not ok:
            summary["error"] = str(result.get("error") or "")[:200]
        self.turn["tools"].append(summary)
        log.info("screen %s: tool %s -> %s", self.screen.display, name, "ok" if ok else result.get("error"))
        output = json.dumps(result, default=str)
        if len(output) > MAX_TOOL_OUTPUT:
            output = json.dumps({"ok": ok, "result": output[:MAX_TOOL_OUTPUT], "truncated": True})
        await ws.send(json.dumps({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": call_id, "output": output}}))
        self.last_activity = time.monotonic()
        return result

    async def _continue_after(self, ws, call_ids):
        try:
            tasks = [self.tool_tasks[c] for c in call_ids if c in self.tool_tasks]
            await asyncio.gather(*tasks, return_exceptions=True)
            for c in call_ids:
                self.tool_tasks.pop(c, None)
            if not self.reason:
                await ws.send(json.dumps({"type": "response.create"}))
        except Exception:
            log.exception("screen %s: could not continue after tools", self.screen.display)
        finally:
            self.followups -= 1
            self.last_activity = time.monotonic()

    # --- the record for the supervisor ------------------------------------------------

    def _new_turn(self):
        return {"user": "", "assistant": "", "tools": []}

    def _publish_turn(self):
        turn, self.turn = self.turn, self._new_turn()
        if not (turn["user"] or turn["assistant"] or turn["tools"]):
            return
        try:
            events.publish(EVENT_TOPIC, {
                "screen": self.screen.display, "ip": self.screen.ip, "sid": self.sid,
                "mode": self.mode, "model": self.cfg.model, "turn": self.turns,
                "at": datetime.now(timezone.utc).isoformat(), **turn})
        except Exception:
            log.exception("could not publish the conversation turn")

