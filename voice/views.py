"""Settings → Voice: connected screens and recent detections (admins only)."""
from datetime import datetime, timezone

from bdh_sdk.auth import is_admin
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.utils import timezone as dj_timezone
from django.views.decorators.http import require_GET

from . import clips, config
from .detector import WAKE_WORDS
from .listen import screens


def _admin_only(request):
    if not is_admin(request):
        raise Http404


def _tz():
    try:
        from bdh_sdk.location import tzinfo

        tz = tzinfo()
    except Exception:
        tz = timezone.utc
    if tz is timezone.utc:
        tz = dj_timezone.get_default_timezone()
    return tz


def _fmt(when):
    hour = when.hour % 12 or 12
    return f"{when:%b} {when.day}, {hour}:{when:%M:%S} {'AM' if when.hour < 12 else 'PM'}"


def status_data():
    tz = _tz()

    def local(iso):
        try:
            return _fmt(datetime.fromisoformat(iso).astimezone(tz))
        except (TypeError, ValueError):
            return ""

    cfg = config.current()
    live = screens()
    for s in live:
        s["since_local"] = local(s["since"])
    found = []
    for d in clips.detections():
        d = dict(d)
        d["at_local"] = local(d.get("at"))
        d["label"] = WAKE_WORDS.get(d.get("word"), ("", d.get("word", "")))[1]
        d["has_clip"] = bool(clips.clip_path(d.get("clip")))
        found.append(d)
    return {
        "listen": cfg.listen,
        "word": cfg.word,
        "word_label": WAKE_WORDS[cfg.word][1],
        "threshold": cfg.threshold,
        "save_clips": cfg.save_clips,
        "timezone": str(tz),
        "screens": live,
        "detections": found,
    }


@require_GET
def status_page(request):
    _admin_only(request)
    return render(request, "voice/settings_voice.html",
                  {"active": "voice", "data": status_data()})


@require_GET
def status_json(request):
    _admin_only(request)
    return JsonResponse(status_data())


@require_GET
def clip(request, name):
    _admin_only(request)
    path = clips.clip_path(name)
    if path is None:
        raise Http404
    return FileResponse(path.open("rb"), content_type="audio/wav", filename=name)
