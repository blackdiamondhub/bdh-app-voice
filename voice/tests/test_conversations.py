"""The conversation record (voice.conversations): storage, retention, recovery, replay, pages."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from bdh_sdk import rpc
from bdh_sdk.testing import FakeBroker
from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from voice import config, conversations, dataset
from voice.plugin import spec
from voice.tests.test_voice import _DataDir

T0 = datetime(2026, 12, 1, 18, 0, tzinfo=timezone.utc)


def record(rec, conversation_id, started=T0, stay="none", checkout="", turns=1, finish=True,
           summary=None):
    """Write a conversation through the recorder as the assistant would."""
    rec.started(conversation_id, {"screen": "lobby", "ip": "192.168.10.50", "sid": "s-" + conversation_id,
                                  "mode": "guest", "model": "m", "stay": stay, "stay_key": "k",
                                  "stay_checkout": checkout, "started_at": started.isoformat()})
    for seq in range(1, turns + 1):
        rec.turn(conversation_id, {"seq": seq, "at": (started + timedelta(seconds=seq)).isoformat(),
                                   "user": f"question {seq}", "assistant": f"answer {seq}",
                                   "tools": [{"name": "end_conversation", "args": {}, "ok": True}]},
                 {"screen": "lobby", "ip": "192.168.10.50", "sid": "s-" + conversation_id,
                  "mode": "guest", "model": "m"})
    if finish:
        rec.finished(conversation_id, {"started_at": started.isoformat(),
                                       "ended_at": (started + timedelta(seconds=30)).isoformat(),
                                       "end_reason": "idle", "seconds": 30.0, "turn_count": turns,
                                       "user_turns": turns, "replies": turns, "tool_calls": turns,
                                       "missing": [], **(summary or {})})
    assert rec.flush()


class _Broker(_DataDir):
    def setUp(self):
        super().setUp()
        self.broker = FakeBroker.for_plugin(spec, install=True)
        self.broker.start()
        self.addCleanup(self.broker.stop)

    def finals(self):
        self.assertTrue(self.recorder.flush())
        rpc.flush()
        return [p["data"] for p in self.broker.ops("events.publish")
                if p["topic"] == conversations.TOPIC and p["data"].get("kind") == "final"]


class RecordTests(_Broker, SimpleTestCase):
    def test_a_conversation_is_stored_then_published(self):
        record(self.recorder, "c1", turns=2)
        rec = conversations.get("c1")
        self.assertEqual([t["seq"] for t in rec["turns"]], [1, 2])
        self.assertEqual((rec["completeness"], rec["final"], rec["turn_count"], rec["tool_calls"]),
                         ("complete", True, 2, 2))
        [final] = self.finals()
        self.assertEqual((final["conversation_id"], final["completeness"], final["replay"]),
                         ("c1", "complete", False))
        self.assertEqual([t["user"] for t in final["turns"]], ["question 1", "question 2"])

    def test_a_missing_turn_makes_the_record_partial(self):
        record(self.recorder, "c1", turns=1, summary={"turn_count": 2})
        rec = conversations.get("c1")
        self.assertEqual(rec["completeness"], "partial")
        self.assertIn("turns_missing", rec["missing"])

    def test_a_reported_gap_makes_the_record_partial(self):
        record(self.recorder, "c1", summary={"missing": ["too_many_turns"]})
        self.assertEqual(conversations.get("c1")["completeness"], "partial")

    def test_an_unfinished_conversation_is_recovered_as_unknown(self):
        record(self.recorder, "c1", turns=2, finish=False)
        self.assertFalse(conversations.get("c1")["final"])
        # The app restarted: a new recorder doesn't know c1 is open.
        rec = conversations.Recorder(settings=lambda: self.cfg)
        with patch.object(conversations, "recorder", rec):
            self.assertTrue(rec.flush())
            rpc.flush()
        c1 = conversations.get("c1")
        self.assertEqual((c1["completeness"], c1["end_reason"], c1["final"], c1["turn_count"]),
                         ("unknown", "lost", True, 2))
        self.assertIn("app_restarted", c1["missing"])
        self.assertEqual([f["conversation_id"] for f in self.finals()], ["c1"])

    def test_open_conversations_are_not_recovered(self):
        record(self.recorder, "c1", finish=False)
        self.recorder._pruned_at = None
        self.recorder._housekeeping()
        self.assertFalse(conversations.get("c1")["final"])

    def test_a_big_record_is_trimmed_for_the_bus_and_marked_partial(self):
        rec = {"id": "c1", "completeness": "complete", "missing": [],
               "turns": [{"seq": i, "user": "u" * 4000, "assistant": "a" * 4000,
                          "tools": [{"name": "t", "ok": True, "result": "r" * 200}]}
                         for i in range(1, 60)]}
        data = conversations.final_event(rec)
        self.assertLess(len(str(data)), conversations.MAX_EVENT_BYTES)
        self.assertEqual(data["completeness"], "partial")
        self.assertIn("event_trimmed", data["missing"])
        self.assertEqual(len(data["turns"]), 59)
        # The stored record keeps everything.
        self.assertEqual(len(rec["turns"][0]["user"]), 4000)


class RetentionTests(_Broker, SimpleTestCase):
    def test_guest_conversations_are_kept_until_checkout_plus_days(self):
        checkout = T0 + timedelta(days=3)
        row = {"stay": "guest", "stay_checkout": checkout.isoformat(), "started_at": T0.isoformat()}
        self.assertEqual(conversations.expires_at(row, guest_days=7, days=30), checkout + timedelta(days=7))

    def test_other_conversations_are_kept_days_after_they_started(self):
        for stay in ("owner", "none", "unknown"):
            row = {"stay": stay, "stay_checkout": "", "started_at": T0.isoformat()}
            self.assertEqual(conversations.expires_at(row, 7, 30), T0 + timedelta(days=30))
        # A guest stay without a known checkout falls back to the same.
        row = {"stay": "guest", "stay_checkout": "", "started_at": T0.isoformat()}
        self.assertEqual(conversations.expires_at(row, 7, 30), T0 + timedelta(days=30))

    def test_prune_deletes_expired_records_and_their_turns(self):
        checkout = T0 + timedelta(days=2)
        record(self.recorder, "guest", stay="guest", checkout=checkout.isoformat())
        record(self.recorder, "owner", stay="owner")
        record(self.recorder, "between", stay="none", started=T0 + timedelta(days=20))
        # 8 days after checkout: the guest conversation goes, the others stay.
        self.assertEqual(conversations.prune(checkout + timedelta(days=8), 7, 30), 1)
        self.assertIsNone(conversations.get("guest"))
        self.assertEqual(conversations._query("SELECT COUNT(*) AS n FROM turns "
                                              "WHERE conversation_id = 'guest'")[0]["n"], 0)
        # 31 days after it started, the owner conversation goes too.
        self.assertEqual(conversations.prune(T0 + timedelta(days=31), 7, 30), 1)
        self.assertEqual([r["id"] for r in conversations.rows()], ["between"])

    def test_the_recorder_prunes_with_the_settings(self):
        record(self.recorder, "old", started=datetime.now(timezone.utc) - timedelta(days=3))
        self.cfg = config.Config(transcript_days=2)
        self.recorder._pruned_at = None
        self.recorder._housekeeping()
        self.assertIsNone(conversations.get("old"))


class SyncTests(_Broker, SimpleTestCase):
    def test_sync_replays_final_records_since(self):
        now = datetime.now(timezone.utc)
        record(self.recorder, "old", started=now - timedelta(hours=5))
        record(self.recorder, "new", started=now - timedelta(minutes=10))
        record(self.recorder, "open", started=now, finish=False)
        before = len(self.finals())
        conversations.on_sync(SimpleNamespace(data={"since": (now - timedelta(hours=1)).isoformat()}))
        replayed = self.finals()[before:]
        self.assertEqual([f["conversation_id"] for f in replayed], ["new"])
        self.assertTrue(replayed[0]["replay"])

    def test_sync_without_since_is_capped(self):
        record(self.recorder, "ancient", started=datetime.now(timezone.utc) - timedelta(days=20))
        record(self.recorder, "recent", started=datetime.now(timezone.utc) - timedelta(days=1))
        before = len(self.finals())
        conversations.on_sync(SimpleNamespace(data={}))
        self.assertEqual([f["conversation_id"] for f in self.finals()[before:]], ["recent"])

    def test_the_app_may_subscribe_to_sync(self):
        self.assertIn("events.subscribe:sentinel/sync", spec.permissions)
        conversations._subscription = None
        self.addCleanup(setattr, conversations, "_subscription", None)
        conversations.subscribe()
        self.assertIsNotNone(conversations._subscription)


class StayInfoTests(SimpleTestCase):
    def test_stay_kinds(self):
        def check(current=None, error=None):
            stays = dataset.StayCheck()
            with patch("bdh_sdk.stays.current", side_effect=error, return_value=current):
                return stays.info()

        def stay(owner):
            return SimpleNamespace(key="abc", stay=SimpleNamespace(
                owner=owner, checkout=datetime(2026, 12, 8, 18, tzinfo=timezone.utc)))

        self.assertEqual(check(None)["stay"], "none")
        self.assertEqual(check(stay(False)), {"stay": "guest", "stay_key": "abc",
                                              "stay_checkout": "2026-12-08T18:00:00+00:00"})
        self.assertEqual(check(stay(True))["stay"], "owner")
        self.assertEqual(check(error=RuntimeError("down"))["stay"], "unknown")

    def test_private_unless_owners_or_nobody(self):
        self.assertTrue(conversations.private("guest"))
        self.assertTrue(conversations.private("unknown"))
        self.assertFalse(conversations.private("owner"))
        self.assertFalse(conversations.private("none"))


class PageTests(_DataDir, TestCase):
    def setUp(self):
        super().setUp()
        broker = FakeBroker.for_plugin(spec, install=True)
        broker.start()
        self.addCleanup(broker.stop)
        record(self.recorder, "c1", turns=2)

    def test_admins_see_the_list_and_a_conversation(self):
        admin = User.objects.create_user("admin", password="x", is_staff=True)
        self.client.force_login(admin)
        page = self.client.get(reverse("voice_conversations"))
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, reverse("voice_conversation", args=["c1"]))
        self.assertContains(page, reverse("voice_status"))  # back
        detail = self.client.get(reverse("voice_conversation", args=["c1"]))
        self.assertContains(detail, "question 2")
        self.assertContains(detail, "end_conversation")
        self.assertContains(detail, reverse("voice_conversations"))  # back
        self.assertEqual(self.client.get(reverse("voice_conversation", args=["nope"])).status_code, 404)

    def test_others_get_nothing(self):
        for user in (None, User.objects.create_user("guest", password="x")):
            if user:
                self.client.force_login(user)
            self.assertEqual(self.client.get(reverse("voice_conversations")).status_code, 404)
            self.assertEqual(self.client.get(reverse("voice_conversation", args=["c1"])).status_code, 404)
