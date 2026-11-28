"""The voice assistant against a fake realtime server and the SDK's FakeBroker."""
import asyncio
import base64
import json
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from bdh_sdk import rpc
from bdh_sdk.plugins import DASHBOARD, GUEST
from bdh_sdk.testing import FakeBroker
from bdh_sdk.tools import Tool, ToolError
from django.test import SimpleTestCase
from websockets.asyncio.server import serve

from voice import assistant, config, conversations, dataset, listen
from voice.audio import Resampler
from voice.detector import CHUNK, SAMPLE_RATE, read_wav
from voice.plugin import spec
from voice.tests.test_voice import FakeDetector, _DataDir, frames, scope

FIXTURES = Path(__file__).resolve().parent / "fixtures"
KEY = "test-key-not-real"
TIMEOUT = 15


def tone(seconds, amp=3000):
    t = np.arange(int(SAMPLE_RATE * seconds))
    return (amp * np.sin(2 * np.pi * 440 * t / SAMPLE_RATE)).astype(np.int16)


async def until(pred, timeout=TIMEOUT, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class FakeRealtime:
    """A realtime API on localhost: ``session.created`` → ``session.updated`` → ``script``."""

    def __init__(self, script=None, refuse=False):
        self.script = script
        self.refuse = refuse
        self.received = []
        self.headers = []
        self.paths = []
        self.connections = 0
        self.closed = 0

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0, max_size=None)
        self.url = f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()

    def events(self, kind):
        return [e for e in self.received if e.get("type") == kind]

    async def wait_for(self, kind, n=1):
        await until(lambda: len(self.events(kind)) >= n, what=f"{n} x {kind}")
        return self.events(kind)[n - 1]

    async def send(self, ws, kind, **fields):
        await ws.send(json.dumps({"type": kind, **fields}))

    async def handler(self, ws):
        self.connections += 1
        self.headers.append(ws.request.headers)
        self.paths.append(ws.request.path)

        async def read():
            async for raw in ws:
                self.received.append(json.loads(raw))

        reader = asyncio.ensure_future(read())
        try:
            await self.send(ws, "session.created", session={})
            await self.wait_for("session.update")
            if self.refuse:
                await self.send(ws, "error", error={"type": "invalid_request_error",
                                                    "code": "bad_voice", "message": "nope"})
                await ws.wait_closed()
                return
            await self.send(ws, "session.updated", session={})
            if self.script:
                await self.script(self, ws)
            await ws.wait_closed()
        finally:
            reader.cancel()
            self.closed += 1

    def appended_samples(self):
        return sum(len(base64.b64decode(e["audio"])) // 2
                   for e in self.events("input_audio_buffer.append"))


class Kiosk:
    """A screen on the listen socket, driven like an ASGI server would."""

    def __init__(self, ip="192.168.10.50"):
        self.ip = ip
        self.sent = []
        self.incoming = asyncio.Queue()

    async def __aenter__(self):
        self.incoming.put_nowait({"type": "websocket.connect"})

        async def receive():
            return await self.incoming.get()

        async def send(message):
            self.sent.append(message)

        query = f"display={self.ip}&v=1.13".encode()
        self.task = asyncio.ensure_future(listen.listen(scope(ip=self.ip, query=query), receive, send))
        await until(lambda: self.texts(), what="hello")
        return self

    async def __aexit__(self, *exc):
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, TIMEOUT)

    def push(self, samples):
        for chunk in frames(samples):
            self.incoming.put_nowait({"type": "websocket.receive", "bytes": chunk})

    def say(self, payload):
        self.incoming.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    def texts(self, kind=None):
        msgs = [json.loads(m["text"]) for m in self.sent if m.get("text")]
        return [m for m in msgs if kind is None or m["type"] == kind]

    def audio(self):
        return [m["bytes"] for m in self.sent if m.get("bytes") is not None]

    async def wait_for(self, kind, n=1):
        await until(lambda: len(self.texts(kind)) >= n, what=f"{n} x {kind} on the kiosk")
        return self.texts(kind)[n - 1]


def fireplace_tool(**kw):
    return Tool("fireplace_on", "Fireplace on", "Turn the fireplace on.",
                params={"type": "object",
                        "properties": {"level": {"type": "integer", "minimum": 1, "maximum": 5}},
                        "required": ["level"]},
                modes=frozenset({GUEST, DASHBOARD}), **kw)


def fc_item(call_id="c1", name="fireplace_on", arguments='{"level": 3}'):
    return {"id": "fc_" + call_id, "type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments}


async def ask_tool(rt, ws, item, n=1):
    """The user asks; the model answers with a function call; returns the tool output."""
    await rt.send(ws, "input_audio_buffer.speech_started", item_id=f"u{n}")
    await rt.send(ws, "input_audio_buffer.speech_stopped", item_id=f"u{n}")
    await rt.send(ws, "conversation.item.input_audio_transcription.completed", item_id=f"u{n}",
                  content_index=0, transcript="Turn on the fireplace")
    await rt.send(ws, "response.created", response={"id": f"r{n}"})
    await rt.send(ws, "response.output_item.done", response_id=f"r{n}", item=item)
    await rt.send(ws, "response.done", response={"id": f"r{n}", "status": "completed",
                                                  "output": [item]})
    out = await rt.wait_for("conversation.item.create", n)
    await rt.wait_for("response.create", n)
    return json.loads(out["item"]["output"])


async def reply(rt, ws, rid, item_id, text, pcm_bytes, status="completed"):
    await rt.send(ws, "response.created", response={"id": rid})
    for i in range(0, len(pcm_bytes), 24000):
        await rt.send(ws, "response.output_audio.delta", response_id=rid, item_id=item_id,
                      delta=base64.b64encode(pcm_bytes[i:i + 24000]).decode())
    await rt.send(ws, "response.output_audio_transcript.done", response_id=rid, item_id=item_id,
                  transcript=text)
    await rt.send(ws, "response.output_audio.done", response_id=rid, item_id=item_id)
    await rt.send(ws, "response.done", response={"id": rid, "status": status, "output": [
        {"id": item_id, "type": "message"}]})


class _Assistant(_DataDir):
    """Fake detector that wakes at chunk 3, a FakeBroker with a key and a fireplace tool."""

    hits = (3,)

    def setUp(self):
        super().setUp()
        p = patch("voice.listen.Detector", side_effect=lambda word: FakeDetector(self.hits))
        p.start()
        self.addCleanup(p.stop)
        self.tool_fn = lambda args: {"level": args["level"], "on": True}
        self.broker = FakeBroker.for_plugin(spec, install=True, **self.broker_kwargs())
        self.broker.start()
        self.addCleanup(self.broker.stop)
        self.broker.secrets[("voice", "api_key")] = KEY
        self.broker.add_tool(fireplace_tool(), lambda args: self.tool_fn(args))
        self.addCleanup(assistant._active.clear)

    def broker_kwargs(self):
        return {}

    def use(self, rt, **cfg):
        cfg.setdefault("follow_up_seconds", 1)
        cfg.setdefault("max_seconds", 30)
        self.cfg = config.Config(endpoint=rt.url, **cfg)

    def published(self, kind="turn"):
        self.assertTrue(self.recorder.flush())
        rpc.flush()
        return [p["data"] for p in self.broker.ops("events.publish")
                if p["topic"] == assistant.EVENT_TOPIC and p["data"].get("kind") == kind]


class ConversationTests(_Assistant, SimpleTestCase):
    def test_full_conversation(self):
        long_reply = np.full(48000 * 2, 1000, dtype="<i2").tobytes()[:96000]  # 2 s

        async def script(rt, ws):
            await rt.wait_for("input_audio_buffer.append")
            self.tool_output = await ask_tool(rt, ws, fc_item())
            await reply(rt, ws, "r2", "a1", "The fireplace is on.", b"\x01\x00" * 9600)
            await asyncio.sleep(0.1)
            # A long reply, interrupted 0.4 s in.
            await rt.send(ws, "response.created", response={"id": "r3"})
            await rt.send(ws, "response.output_audio.delta", response_id="r3", item_id="a2",
                          delta=base64.b64encode(long_reply).decode())
            await asyncio.sleep(0.4)
            await rt.send(ws, "input_audio_buffer.speech_started", item_id="u2")
            await rt.wait_for("response.cancel")
            await rt.send(ws, "error", error={"code": "response_cancel_not_active"})
            await rt.send(ws, "response.done", response={"id": "r3", "status": "cancelled",
                                                          "output": []})
            # Audio from the cancelled reply must not reach the screen.
            await rt.send(ws, "response.output_audio.delta", response_id="r3", item_id="a2",
                          delta=base64.b64encode(b"\x00" * 4800).decode())
            await rt.send(ws, "input_audio_buffer.speech_stopped", item_id="u2")
            # ...then silence: the conversation ends after follow_up_seconds.

        async def go():
            async with FakeRealtime(script) as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(1.0))  # wake at chunk 3 = 3840 samples
                    await kiosk.wait_for("state")
                    kiosk.push(tone(0.5))
                    end = await kiosk.wait_for("conversation_end")
                    self.assertEqual(end, {"type": "conversation_end", "reason": "idle"})
                    await until(lambda: rt.closed, what="realtime socket closed")
                    return rt, kiosk

        with self.assertLogs("voice.assistant", "INFO") as logs:
            rt, kiosk = asyncio.run(go())
        # Both sides of the conversation are in the log, so a reply can be checked later.
        self.assertIn("INFO:voice.assistant:screen 192.168.10.50: user: Turn on the fireplace",
                      logs.output)
        self.assertIn("INFO:voice.assistant:screen 192.168.10.50: assistant: The fireplace is on.",
                      logs.output)

        # The model was set up for the lodge with the gateway's tools, and authenticated.
        self.assertEqual(rt.headers[0]["api-key"], KEY)
        self.assertEqual(rt.paths[0], "/openai/v1/realtime?model=gpt-realtime-2.1-mini")
        session = rt.events("session.update")[0]["session"]
        self.assertEqual(session["type"], "realtime")
        self.assertIn("Black Diamond Lodge", session["instructions"])
        self.assertEqual([t["name"] for t in session["tools"]], ["end_conversation", "fireplace_on"])
        self.assertEqual(session["tools"][1]["parameters"]["required"], ["level"])
        self.assertEqual(session["audio"]["output"]["voice"], "marin")
        self.assertEqual(session["audio"]["input"]["format"], {"type": "audio/pcm", "rate": 24000})
        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "semantic_vad")

        # Mic audio after the wake word, resampled to 24 kHz.
        after_wake = int(1.5 * SAMPLE_RATE) - 3 * CHUNK
        self.assertAlmostEqual(rt.appended_samples(), after_wake * 1.5, delta=3)

        # The tool ran through the gateway with the model's arguments.
        self.assertEqual(len(self.broker.tool_calls), 1)
        self.assertEqual(self.broker.tool_calls[0]["args"], {"level": 3})
        self.assertEqual(list(self.broker.sessions.values())[0]["display"], "192.168.10.50")
        self.assertEqual(self.tool_output, {"ok": True, "result": {"level": 3, "on": True}})
        self.assertEqual(rt.events("conversation.item.create")[0]["item"]["call_id"], "c1")

        # Reply audio reached the screen, framed, and stopped at the barge-in.
        kinds = [m["type"] for m in kiosk.texts()]
        self.assertEqual(kinds[:3], ["hello", "wake", "state"])
        self.assertIn("speak_start", kinds)
        self.assertLess(kinds.index("speak_start"), kinds.index("speak_end"))
        self.assertLess(kinds.index("speak_end"), kinds.index("speak_stop"))
        chunks = kiosk.audio()
        self.assertEqual(sum(map(len, chunks)), 19200 + 96000)
        self.assertTrue(all(len(c) <= assistant.AUDIO_FRAME_BYTES for c in chunks))
        states = [m["state"] for m in kiosk.texts("state")]
        self.assertEqual(states[0], "listening")
        self.assertIn("thinking", states)
        self.assertIn("speaking", states)
        self.assertEqual(states[-1], "idle")
        finals = [(m["role"], m["text"]) for m in kiosk.texts("transcript") if m["final"]]
        self.assertIn(("user", "Turn on the fireplace"), finals)
        self.assertIn(("assistant", "The fireplace is on."), finals)

        # Barge-in: the played part of the reply was kept, the rest cancelled.
        trunc = rt.events("conversation.item.truncate")[0]
        self.assertEqual(trunc["item_id"], "a2")
        self.assertGreaterEqual(trunc["audio_end_ms"], 300)
        self.assertLess(trunc["audio_end_ms"], 2000)
        self.assertEqual(len(rt.events("response.cancel")), 1)

        # The turn was stored, then published for the sentinel (not on assistant/...).
        turns = self.published()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user"], "Turn on the fireplace")
        self.assertEqual(turns[0]["assistant"], "The fireplace is on.")
        self.assertEqual(turns[0]["tools"], [{
            "name": "fireplace_on", "args": {"level": 3}, "ok": True,
            "result": json.dumps({"ok": True, "result": {"level": 3, "on": True}})}])
        self.assertEqual(turns[0]["screen"], "192.168.10.50")
        self.assertTrue(turns[0]["sid"])
        self.assertEqual(turns[0]["seq"], 1)
        self.assertEqual(assistant._active, {})

        # ...and the whole conversation once it ended.
        [final] = self.published("final")
        self.assertEqual(final["conversation_id"], turns[0]["conversation_id"])
        self.assertEqual((final["completeness"], final["end_reason"], final["turn_count"],
                          final["tool_calls"], final["stay"], final["mode"]),
                         ("complete", "idle", 1, 1, "none", "dashboard"))
        self.assertEqual([t["seq"] for t in final["turns"]], [1])
        self.assertEqual(conversations.get(final["conversation_id"])["turns"][0]["user"],
                         "Turn on the fireplace")

        # The wake's training clip learned how the conversation went.
        self.assertTrue(self.writer.flush())
        [clip] = dataset.rows()
        self.assertEqual((clip["fired"], clip["end_reason"], clip["user_turns"], clip["tool_calls"]),
                         (1, "idle", 1, 1))
        self.assertEqual(clip["transcript"], "Turn on the fireplace")
        self.assertEqual(clip["auto_label"], dataset.LIKELY_TRUE)

    def test_tool_error_goes_back_to_the_model(self):
        def broken(args):
            raise ToolError("the fireplace is offline")
        self.tool_fn = broken

        async def script(rt, ws):
            self.tool_output = await ask_tool(rt, ws, fc_item())
            bad = await ask_tool(rt, ws, fc_item("c2", arguments="{nope"), n=2)
            self.bad_json = bad

        async def go():
            async with FakeRealtime(script) as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    await kiosk.wait_for("conversation_end")

        asyncio.run(go())
        self.assertEqual(self.tool_output, {"ok": False, "error": "the fireplace is offline"})
        self.assertFalse(self.bad_json["ok"])
        self.assertEqual(len(self.broker.tool_calls), 1)
        tools_used = [t for turn in self.published() for t in turn["tools"]]
        self.assertEqual(tools_used[0]["error"], "the fireplace is offline")

    def test_saying_thats_all_ends_after_the_goodbye(self):
        goodbye = b"\x01\x00" * 24000  # 1 s of audio

        async def script(rt, ws):
            await rt.wait_for("input_audio_buffer.append")
            # "Oh wait" after an end call keeps the conversation going...
            await rt.send(ws, "response.created", response={"id": "r0"})
            await rt.send(ws, "response.output_item.done", response_id="r0",
                          item=fc_item("c0", "end_conversation", "{}"))
            await rt.send(ws, "input_audio_buffer.speech_started", item_id="u0")
            await rt.send(ws, "response.done", response={"id": "r0", "status": "completed", "output": [
                fc_item("c0", "end_conversation", "{}")]})
            await rt.send(ws, "input_audio_buffer.speech_stopped", item_id="u0")
            await asyncio.sleep(0.5)
            self.assertEqual(kiosk.texts("conversation_end"), [])
            # ..."that's all, thanks": a spoken goodbye and the end call in one reply.
            await rt.send(ws, "input_audio_buffer.speech_started", item_id="u1")
            await rt.send(ws, "input_audio_buffer.speech_stopped", item_id="u1")
            self.said_bye = time.monotonic()
            item = fc_item("c1", "end_conversation", "{}")
            await rt.send(ws, "response.created", response={"id": "r1"})
            await rt.send(ws, "response.output_audio.delta", response_id="r1", item_id="a1",
                          delta=base64.b64encode(goodbye).decode())
            await rt.send(ws, "response.output_audio.done", response_id="r1", item_id="a1")
            await rt.send(ws, "response.output_item.done", response_id="r1", item=item)
            await rt.send(ws, "response.done", response={"id": "r1", "status": "completed", "output": [
                {"id": "a1", "type": "message"}, item]})

        async def go():
            nonlocal kiosk
            async with FakeRealtime(script) as rt:
                self.use(rt, follow_up_seconds=30)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    end = await kiosk.wait_for("conversation_end")
                    ended = time.monotonic()
                    return rt, end, ended

        kiosk = None
        rt, end, ended = asyncio.run(go())
        self.assertEqual(end, {"type": "conversation_end", "reason": "done"})
        # Not cut off: the screen got the whole goodbye, and the end waited for it to play.
        self.assertEqual(sum(map(len, kiosk.audio())), len(goodbye))
        self.assertEqual(kiosk.texts("speak_stop"), [])
        self.assertGreaterEqual(ended - self.said_bye, 1.0)
        self.assertLess(ended - self.said_bye, 5)
        # Handled here: no gateway call, no tool output, no follow-up reply.
        self.assertEqual(self.broker.tool_calls, [])
        self.assertEqual(rt.events("conversation.item.create"), [])
        self.assertEqual(rt.events("response.create"), [])
        # ...but end_conversation counts as a tool call in the record.
        [final] = self.published("final")
        self.assertEqual(final["tool_calls"], 2)
        self.assertEqual([[t["name"] for t in turn["tools"]] for turn in final["turns"]],
                         [["end_conversation"], ["end_conversation"]])
        self.assertEqual([t["seq"] for t in final["turns"]], [1, 2])

    def test_wake_detection_paused_during_a_conversation(self):
        self.hits = (3, 40)  # 40 is 3.2 s in, well past the debounce

        async def go():
            async with FakeRealtime() as rt:
                self.use(rt, follow_up_seconds=30)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(4.0))
                    await until(lambda: rt.appended_samples() >= (4 * SAMPLE_RATE - 3 * CHUNK) * 1.5 - 3,
                                what="all audio forwarded")
                    self.assertEqual(len(kiosk.texts("wake")), 1)
                    kiosk.say({"type": "end"})
                    end = await kiosk.wait_for("conversation_end")
                    self.assertEqual(end["reason"], "ended")

        asyncio.run(go())

    def test_max_seconds(self):
        async def go():
            async with FakeRealtime() as rt:
                self.use(rt, follow_up_seconds=30, max_seconds=1)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    started = time.monotonic()
                    end = await kiosk.wait_for("conversation_end")
                    self.assertEqual(end["reason"], "max_seconds")
                    self.assertLess(time.monotonic() - started, 5)

        asyncio.run(go())

    def test_disconnect_ends_the_conversation(self):
        async def go():
            async with FakeRealtime() as rt:
                self.use(rt, follow_up_seconds=30)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    await rt.wait_for("input_audio_buffer.append")
                await until(lambda: rt.closed, what="realtime socket closed")
                self.assertEqual(assistant._active, {})

        asyncio.run(go())

    def test_max_concurrency(self):
        async def go():
            async with FakeRealtime() as rt:
                self.use(rt, follow_up_seconds=30)
                with patch.object(assistant, "MAX_CONCURRENT", 1):
                    async with Kiosk("192.168.10.50") as a, Kiosk("192.168.10.51") as b:
                        a.push(tone(0.5))
                        await rt.wait_for("input_audio_buffer.append")
                        b.push(tone(0.5))
                        end = await b.wait_for("conversation_end")
                        self.assertEqual(end["reason"], "busy")
                        self.assertEqual(rt.connections, 1)
                        a.say({"type": "end"})
                        await a.wait_for("conversation_end")

        asyncio.run(go())

    def test_realtime_refusal_ends_with_a_message(self):
        async def go():
            async with FakeRealtime(refuse=True) as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    return await kiosk.wait_for("conversation_end")

        end = asyncio.run(go())
        self.assertEqual(end["reason"], "error")
        self.assertTrue(end["message"])


class RecordTests(_Assistant, SimpleTestCase):
    """What the log and the record keep, depending on who is staying."""

    SAID = "it's freezing, put the fireplace on"
    REPLY = "The fireplace is on."

    def talk(self):
        async def script(rt, ws):
            await rt.send(ws, "conversation.item.input_audio_transcription.completed", item_id="u1",
                          content_index=0, transcript=self.SAID)
            await reply(rt, ws, "r1", "a1", self.REPLY, b"\x01\x00" * 480)

        async def go():
            async with FakeRealtime(script) as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    await kiosk.wait_for("conversation_end")

        with self.assertLogs("voice", "INFO") as logs:
            asyncio.run(go())
        return "\n".join(logs.output)

    def test_guest_stay_keeps_the_words_out_of_the_log(self):
        self.stays.guest = True
        out = self.talk()
        self.assertNotIn("freezing", out)
        self.assertNotIn(self.REPLY, out)
        self.assertIn(f"user: ({len(self.SAID)} chars, conversation ", out)
        self.assertIn(f"assistant: ({len(self.REPLY)} chars, conversation ", out)
        # The record (admins only) still has them.
        [final] = self.published("final")
        self.assertEqual(final["stay"], "guest")
        self.assertEqual(final["turns"][0]["user"], self.SAID)
        self.assertEqual(final["completeness"], "complete")

    def test_unknown_stay_is_treated_like_a_guest_stay(self):
        with patch("voice.conversations.stay_info", side_effect=RuntimeError("no broker")):
            out = self.talk()
        self.assertNotIn("freezing", out)
        [final] = self.published("final")
        self.assertEqual(final["stay"], "unknown")

    def test_owner_stay_logs_the_words(self):
        owners = {"stay": "owner", "stay_key": "s1", "stay_checkout": ""}
        with patch("voice.conversations.stay_info", return_value=owners):
            out = self.talk()
        self.assertIn(f"user: {self.SAID}", out)
        self.assertIn(f"assistant: {self.REPLY}", out)
        [final] = self.published("final")
        self.assertEqual((final["stay"], final["stay_key"]), ("owner", "s1"))

    def test_between_stays_logs_the_words(self):
        out = self.talk()
        self.assertIn(f"user: {self.SAID}", out)

    def test_a_cut_text_makes_the_record_partial(self):
        self.SAID = "x" * (conversations.MAX_TEXT + 10)
        self.talk()
        [final] = self.published("final")
        self.assertEqual(final["completeness"], "partial")
        self.assertIn("text_cut", final["missing"])
        self.assertEqual(len(final["turns"][0]["user"]), conversations.MAX_TEXT)

    def test_a_failed_write_makes_the_record_partial(self):
        with patch("voice.conversations._insert_turn", side_effect=OSError("disk full")):
            self.talk()
            [final] = self.published("final")
        self.assertEqual(final["completeness"], "partial")
        self.assertIn("write_failed", final["missing"])
        # The turn still reached the bus.
        self.assertEqual(self.published()[0]["seq"], 1)


class GatewayRefusedTests(_Assistant, SimpleTestCase):
    def broker_kwargs(self):
        return {"permissions": ()}  # as if the hub hadn't granted "tools"

    def test_conversation_without_tools(self):
        async def script(rt, ws):
            self.tool_output = await ask_tool(rt, ws, fc_item())

        async def go():
            async with FakeRealtime(script) as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    await kiosk.wait_for("conversation_end")
                    return rt

        rt = asyncio.run(go())
        session = rt.events("session.update")[0]["session"]
        # Only the built-in end_conversation, which never reaches the gateway.
        self.assertEqual([t["name"] for t in session["tools"]], ["end_conversation"])
        self.assertFalse(self.tool_output["ok"])
        self.assertEqual(self.broker.tool_calls, [])


class NotConfiguredTests(_Assistant, SimpleTestCase):
    def test_missing_key(self):
        del self.broker.secrets[("voice", "api_key")]

        async def go():
            async with FakeRealtime() as rt:
                self.use(rt)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(0.5))
                    end = await kiosk.wait_for("conversation_end")
                    self.assertEqual(end["reason"], "not_configured")
                    self.assertEqual(end["message"], assistant.NOT_CONFIGURED)
                    self.assertEqual(len(kiosk.texts("wake")), 1)
                self.assertEqual(rt.connections, 0)

        asyncio.run(go())

    def test_disabled_only_wakes(self):
        async def go():
            async with FakeRealtime() as rt:
                self.use(rt, assistant=False)
                async with Kiosk() as kiosk:
                    kiosk.push(tone(1.0))
                    await kiosk.wait_for("wake")
                    await asyncio.sleep(0.5)
                    self.assertFalse(kiosk.texts("hello")[0]["assistant"])
                    self.assertEqual([m["type"] for m in kiosk.texts()], ["hello", "wake"])
                self.assertEqual(rt.connections, 0)

        asyncio.run(go())


class HelperTests(SimpleTestCase):
    def test_resampler_tracks_the_rate_across_any_chunking(self):
        r = Resampler()
        rng = np.random.default_rng(1)
        n_in = n_out = 0
        for _ in range(500):
            k = int(rng.integers(1, 700))
            n_in += k
            n_out += len(r(np.zeros(k, dtype=np.int16)))
        self.assertAlmostEqual(n_out, n_in * 1.5, delta=2)

    def test_resampler_is_continuous(self):
        r = Resampler()
        ramp = np.arange(3000, dtype=np.int16)
        out = np.concatenate([r(ramp[i:i + 7]) for i in range(0, len(ramp), 7)])
        self.assertLessEqual(np.max(np.abs(np.diff(out.astype(int)))), 1)
        self.assertAlmostEqual(len(Resampler()(np.zeros(320, dtype=np.int16))), 480, delta=1)

    def test_realtime_url(self):
        self.assertEqual(assistant.realtime_url("https://x.openai.azure.com", "m"),
                         "wss://x.openai.azure.com/openai/v1/realtime?model=m")
        self.assertEqual(assistant.realtime_url("x.openai.azure.com/", "m"),
                         "wss://x.openai.azure.com/openai/v1/realtime?model=m")
        self.assertEqual(assistant.realtime_url("http://127.0.0.1:9/openai/v1/realtime", "a b"),
                         "ws://127.0.0.1:9/openai/v1/realtime?model=a+b")

    def test_lodge_time_is_utc_minus_seven(self):
        now = datetime(2026, 12, 24, 20, 5, tzinfo=timezone.utc)
        text = assistant.instructions(config.Config(instructions="Checkout is at 10."), now)
        self.assertIn("Thursday, December 24, 2026, 1:05 PM", text)
        self.assertTrue(text.endswith("Checkout is at 10."))

    def test_screen_room(self):
        cfg = config.Config(screen_rooms="10.0.0.5=the kitchen; 10.0.0.6 = the great room")
        self.assertEqual(config.room_for(cfg, "10.0.0.6", "Lobby"), "the great room")
        self.assertEqual(config.room_for(cfg, "10.0.0.9"), "")
        self.assertEqual(config.room_for(config.Config(screen_rooms="the great room"), "10.0.0.9"),
                         "the great room")
        self.assertIn("This screen is in the great room.",
                      assistant.instructions(cfg, room="the great room"))
        self.assertNotIn("This screen is in", assistant.instructions(cfg))

    def test_load_reads_every_setting(self):
        # The other tests replace config.load, so run the real one end to end.
        stored = {"listen": "false", "threshold": "70", "model": "gpt-realtime-2.1",
                  "screen_rooms": " the great room ", "transcript_guest_days": "3",
                  "transcript_days": "900"}
        with patch("voice.config.app_setting", side_effect=lambda app, key, default: stored.get(key, default)):
            cfg = config.load()
        self.assertEqual((cfg.transcript_guest_days, cfg.transcript_days), (3, 365))
        self.assertFalse(cfg.listen)
        self.assertEqual(cfg.threshold, 0.7)
        self.assertEqual(cfg.model, "gpt-realtime-2.1")
        self.assertEqual(cfg.screen_rooms, "the great room")
        self.assertTrue(cfg.assistant)

    def test_confirm_tools_say_so(self):
        [_, t] = assistant.function_tools([{"name": "a", "title": "Unlock", "description": "Unlock the door.",
                                         "parameters": {"type": "object", "properties": {}},
                                         "confirm": True}])
        self.assertEqual(t["description"], "Unlock: Unlock the door. (confirm first)")


@unittest.skipUnless(os.environ.get("VOICE_LIVE_TEST") and os.environ.get("BDH_VOICE_API_KEY"),
                     "set VOICE_LIVE_TEST=1 and BDH_VOICE_API_KEY to talk to Azure")
class LiveSmokeTest(_Assistant, SimpleTestCase):
    """One short real conversation: asks the time with a recorded question."""

    def test_live_question(self):
        self.broker.secrets[("voice", "api_key")] = os.environ["BDH_VOICE_API_KEY"]
        endpoint = os.environ.get("VOICE_LIVE_ENDPOINT", config.DEFAULT_ENDPOINT)
        self.cfg = config.Config(endpoint=endpoint, follow_up_seconds=3, max_seconds=40,
                                 model=os.environ.get("VOICE_LIVE_MODEL", config.DEFAULT_MODEL))
        question = read_wav(FIXTURES / "question.wav")

        async def go():
            async with Kiosk() as kiosk:
                kiosk.push(np.zeros(3 * CHUNK + 160, dtype=np.int16))  # wake
                await kiosk.wait_for("state")
                for chunk in frames(question):
                    kiosk.incoming.put_nowait({"type": "websocket.receive", "bytes": chunk})
                    await asyncio.sleep(0.02)
                spoke = time.monotonic()
                silence = np.zeros(320, dtype=np.int16).tobytes()
                while not kiosk.audio() and time.monotonic() - spoke < 15:
                    kiosk.incoming.put_nowait({"type": "websocket.receive", "bytes": silence})
                    await asyncio.sleep(0.02)
                latency = time.monotonic() - spoke
                while not kiosk.texts("conversation_end") and time.monotonic() - spoke < 40:
                    kiosk.incoming.put_nowait({"type": "websocket.receive", "bytes": silence})
                    await asyncio.sleep(0.02)
                return kiosk, latency

        kiosk, latency = asyncio.run(go())
        print(f"\nlive: first reply audio {latency:.2f} s after the end of the question")
        print("live:", [(m["role"], m["text"]) for m in kiosk.texts("transcript") if m["final"]])
        self.assertTrue(kiosk.audio(), kiosk.texts())
        self.assertEqual(kiosk.texts("conversation_end")[0]["reason"], "idle")
