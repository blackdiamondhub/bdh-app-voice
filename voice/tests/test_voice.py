import asyncio
import json
import shutil
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from voice import clips, config, listen
from voice.detector import CHUNK, SAMPLE_RATE, Detector, read_wav, score_clip

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LAN = ("127.0.0.1", 50000)
FRAME = 640  # bytes: 20 ms of 16 kHz int16


def frames(samples, size=FRAME):
    data = np.asarray(samples, dtype="<i2").tobytes()
    return [data[i:i + size] for i in range(0, len(data), size)]


def silence(seconds):
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16)


def scope(ip="192.168.10.50", peer=LAN, query=b"display=192.168.10.50&v=1.2.3"):
    headers = [(b"x-forwarded-for", ip.encode())] if ip else []
    return {"type": "websocket", "path": "/voice/api/listen", "query_string": query,
            "headers": headers, "client": peer}


def run_socket(scope_, chunks=()):
    """Drive ``listen`` like an ASGI server; return what it sent."""
    sent = []
    total = sum(len(c) for c in chunks) // 2

    async def go():
        loop = asyncio.get_running_loop()
        incoming = asyncio.Queue()
        incoming.put_nowait({"type": "websocket.connect"})

        async def receive():
            return await incoming.get()

        async def send(message):
            sent.append(message)

        task = asyncio.ensure_future(listen.listen(scope_, receive, send))
        for chunk in chunks:
            incoming.put_nowait({"type": "websocket.receive", "bytes": chunk})
        # Wait until the detector has heard everything (or the socket closed).
        deadline = loop.time() + 60
        while loop.time() < deadline and not task.done():
            live = list(listen._screens.values())
            if live and live[0].heard >= total:
                break
            if not live and any(m["type"] == "websocket.close" for m in sent):
                break
            await asyncio.sleep(0.02)
        incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(task, 30)

    asyncio.run(go())
    return sent


def texts(sent):
    return [json.loads(m["text"]) for m in sent if m["type"] == "websocket.send"]


class DetectorTests(SimpleTestCase):
    def test_hey_jarvis_is_heard(self):
        score = score_clip(read_wav(FIXTURES / "hey_jarvis.wav"))
        self.assertGreater(score, 0.9)

    def test_other_speech_is_not(self):
        self.assertLess(score_clip(read_wav(FIXTURES / "other_speech.wav")), 0.1)

    def test_other_wake_words_ignore_jarvis(self):
        samples = read_wav(FIXTURES / "hey_jarvis.wav")
        self.assertLess(score_clip(samples, "alexa"), 0.1)

    def test_any_frame_size_gives_the_same_scores(self):
        samples = np.concatenate((silence(1), read_wav(FIXTURES / "hey_jarvis.wav"), silence(1)))
        whole = Detector().feed(samples)
        det = Detector()
        odd = []
        for i in range(0, len(samples), 333):
            odd += det.feed(samples[i:i + 333])
        self.assertEqual(len(whole), len(samples) // CHUNK)
        np.testing.assert_allclose(odd, whole[:len(odd)], atol=1e-6)

    def test_unknown_word(self):
        with self.assertRaises(ValueError):
            Detector("ok_computer")


class FakeDetector:
    """Scores 0 except at the given chunk numbers."""

    def __init__(self, hits):
        self.hits = set(hits)
        self.n = 0
        self.pending = 0

    @property
    def buffered(self):
        return self.pending

    def feed(self, pcm):
        total = self.pending + len(pcm)
        out = []
        for _ in range(total // CHUNK):
            self.n += 1
            out.append(0.95 if self.n in self.hits else 0.0)
        self.pending = total % CHUNK
        return out


class _DataDir:
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.settings_ctx = override_settings(VOICE_DATA_DIR=self.tmp)
        self.settings_ctx.enable()
        self.addCleanup(self.settings_ctx.disable)
        config.forget()
        self.addCleanup(config.forget)
        self.cfg = config.Config()
        p = patch("voice.config.load", side_effect=lambda: self.cfg)
        p.start()
        self.addCleanup(p.stop)


class SessionTests(_DataDir, SimpleTestCase):
    def session(self, hits, **cfg):
        screen = listen.Screen("192.168.10.50", "lobby", "1.0")
        s = listen.Session(screen, config.Config(**cfg))
        s.detector = FakeDetector(hits)
        return s

    def test_wake_message(self):
        s = self.session([3])
        wakes = s.process(silence(1).tobytes())
        self.assertEqual(len(wakes), 1)
        self.assertEqual(wakes[0]["type"], "wake")
        self.assertEqual(wakes[0]["word"], "hey_jarvis")
        self.assertEqual(wakes[0]["score"], 0.95)
        self.assertIn("T", wakes[0]["at"])
        self.assertEqual(clips.detections()[0]["screen"], "lobby")

    def test_debounce_within_two_seconds_of_audio(self):
        # Chunks are 80 ms: 3 and 20 are 1.36 s apart, 3 and 29 are 2.08 s.
        s = self.session([3, 4, 20, 29])
        wakes = []
        for chunk in frames(silence(3)):
            wakes += s.process(chunk)
        self.assertEqual(len(wakes), 2)

    def test_below_threshold(self):
        s = self.session([3], threshold=0.99)
        self.assertEqual(s.process(silence(1).tobytes()), [])

    def test_odd_byte_frames_are_buffered(self):
        s = self.session([])
        data = silence(0.5).tobytes()
        for i in range(0, len(data), 333):
            s.process(data[i:i + 333])
        self.assertEqual(s.samples, len(data) // 2)

    def test_clip_saved_one_second_after_wake(self):
        s = self.session([5])
        s.process(silence(2).tobytes())  # wake at 0.4 s, clip due at 1.4 s
        entry = clips.detections()[0]
        self.assertTrue(entry["clip"])
        path = clips.clip_path(entry["clip"])
        self.assertIsNotNone(path)
        with wave.open(str(path)) as f:
            self.assertEqual(f.getframerate(), SAMPLE_RATE)

    def test_no_clip_when_off(self):
        s = self.session([3], save_clips=False)
        s.process(silence(3).tobytes())
        self.assertEqual(clips.detections()[0]["clip"], "")
        self.assertFalse((Path(self.tmp) / "clips").exists())

    def test_level(self):
        self.assertEqual(listen.level_dbfs(silence(0.1)), listen.LEVEL_FLOOR)
        full = np.full(1600, 32767, dtype=np.int16)
        self.assertAlmostEqual(listen.level_dbfs(full), 0.0, places=1)


class ListenSocketTests(_DataDir, SimpleTestCase):
    def test_hello_then_wake_from_real_audio(self):
        audio = np.concatenate((silence(1), read_wav(FIXTURES / "hey_jarvis.wav"), silence(1.5)))
        sent = run_socket(scope(), frames(audio))
        self.assertEqual(sent[0], {"type": "websocket.accept"})
        msgs = texts(sent)
        self.assertEqual(msgs[0], {"type": "hello", "words": ["hey_jarvis"], "threshold": 0.5,
                                   "assistant": True, "ping": listen.PING_SECONDS})
        wakes = [m for m in msgs if m["type"] == "wake"]
        self.assertEqual(len(wakes), 1, msgs)
        self.assertGreater(wakes[0]["score"], 0.5)
        self.assertEqual(clips.detections()[0]["screen"], "192.168.10.50")
        self.assertEqual(listen.screens(), [])

    def test_public_address_refused(self):
        sent = run_socket(scope(ip="8.8.8.8"))
        self.assertEqual(sent, [{"type": "websocket.close", "code": listen.FORBIDDEN}])

    def test_public_peer_refused_even_with_lan_header(self):
        sent = run_socket(scope(peer=("8.8.8.8", 1234)))
        self.assertEqual(sent, [{"type": "websocket.close", "code": listen.FORBIDDEN}])

    def test_direct_lan_peer_accepted(self):
        sent = run_socket(scope(ip="", peer=("10.0.0.7", 1234)))
        self.assertEqual(texts(sent)[0]["type"], "hello")

    def test_listen_off(self):
        self.cfg = config.Config(listen=False)
        sent = run_socket(scope())
        self.assertEqual(sent[0], {"type": "websocket.accept"})
        self.assertEqual(texts(sent), [{"type": "config", "listen": False}])
        self.assertEqual(sent[-1]["type"], "websocket.close")

    def test_other_word_and_threshold_in_hello(self):
        self.cfg = config.Config(word="alexa", threshold=0.7)
        sent = run_socket(scope())
        self.assertEqual(texts(sent)[0], {"type": "hello", "words": ["alexa"], "threshold": 0.7,
                                            "assistant": True, "ping": listen.PING_SECONDS})


    def test_keepalive_pings(self):
        sent = []

        async def go():
            incoming = asyncio.Queue()
            incoming.put_nowait({"type": "websocket.connect"})

            async def send(message):
                sent.append(message)

            task = asyncio.ensure_future(listen.listen(scope(), incoming.get, send))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30
            while loop.time() < deadline and texts(sent).count({"type": "ping"}) < 2:
                await asyncio.sleep(0.01)
            incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait_for(task, 30)

        with patch.object(listen, "PING_SECONDS", 0.05):
            asyncio.run(go())
        msgs = texts(sent)
        self.assertEqual(msgs[0]["ping"], 0.05)
        self.assertEqual(msgs[1:3], [{"type": "ping"}] * 2)


class ClipTests(_DataDir, SimpleTestCase):
    def test_ring_keeps_newest(self):
        ring = clips.Ring(seconds=1, rate=10)
        ring.add(np.arange(4))
        np.testing.assert_array_equal(ring.samples(), [0, 1, 2, 3])
        ring.add(np.arange(4, 13))
        np.testing.assert_array_equal(ring.samples(), np.arange(3, 13))
        ring.add(np.arange(100, 125))
        np.testing.assert_array_equal(ring.samples(), np.arange(115, 125))

    def test_recorder_waits_for_the_tail(self):
        rec = clips.ClipRecorder(before=2, after=1, rate=10)
        self.assertEqual(rec.add(np.arange(25)), [])
        rec.mark("a.wav")
        self.assertEqual(rec.add(np.arange(25, 30)), [])
        done = rec.add(np.arange(30, 36))
        self.assertEqual(len(done), 1)
        name, samples = done[0]
        self.assertEqual(name, "a.wav")
        np.testing.assert_array_equal(samples, np.arange(5, 35))

    def test_recorder_places_clips_inside_a_batch(self):
        rec = clips.ClipRecorder(before=2, after=1, rate=10)
        rec.mark("b.wav", at=40)
        [(name, samples)] = rec.add(np.arange(80))
        np.testing.assert_array_equal(samples, np.arange(20, 50))

    def test_keeps_newest_clips(self):
        for i in range(5):
            clips.save_clip(f"c{i}.wav", silence(0.01), keep=3)
        names = sorted(p.name for p in clips.clips_dir().glob("*.wav"))
        self.assertEqual(names, ["c2.wav", "c3.wav", "c4.wav"])

    def test_bad_names(self):
        with self.assertRaises(ValueError):
            clips.save_clip("../x.wav", silence(0.01))
        self.assertIsNone(clips.clip_path("../detections.json"))

    def test_log_keeps_fifty_newest_first(self):
        for i in range(60):
            clips.record({"id": str(i)})
        log = clips.detections()
        self.assertEqual(len(log), clips.KEEP_DETECTIONS)
        self.assertEqual(log[0]["id"], "59")


class StatusPageTests(_DataDir, TestCase):
    def setUp(self):
        super().setUp()
        self.admin = User.objects.create_user("owner", password="pw", is_staff=True)
        clips.save_clip("1.wav", silence(0.1))
        clips.record({"id": "1", "at": "2026-01-02T03:04:05+00:00", "screen": "lobby",
                      "word": "hey_jarvis", "score": 0.9, "clip": "1.wav"})

    def test_guests_get_404(self):
        for name, args in (("voice_status", ()), ("voice_status_json", ()), ("voice_clip", ("1.wav",))):
            self.assertEqual(self.client.get(reverse(name, args=args)).status_code, 404)

    def test_admin_sees_detections(self):
        self.client.force_login(self.admin)
        page = self.client.get(reverse("voice_status"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, "Last 50 detections")
        data = self.client.get(reverse("voice_status_json")).json()
        self.assertEqual(data["detections"][0]["label"], "Hey Jarvis")
        self.assertTrue(data["detections"][0]["has_clip"])
        self.assertTrue(data["detections"][0]["at_local"])
        clip = self.client.get(reverse("voice_clip", args=("1.wav",)))
        self.assertEqual(clip["Content-Type"], "audio/wav")
        self.assertEqual(self.client.get(reverse("voice_clip", args=("2.wav",))).status_code, 404)


class PluginTests(SimpleTestCase):
    def test_plugin_importable(self):
        from bdh_sdk.testing import assert_plugin_importable

        assert_plugin_importable("voice.plugin")
