"""Settings → Voice (screens and detections) and Settings → Wake-word clips (admins only)."""
from datetime import datetime, timezone
from urllib.parse import urlencode

from bdh_sdk.auth import is_admin
from django.http import FileResponse, Http404, HttpResponseRedirect, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone as dj_timezone
from django.views.decorators.http import require_GET, require_POST

from . import clips, config, dataset
from .assistant import LODGE_UTC_OFFSET
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


# --- wake-word training clips -------------------------------------------------------

PAGE_SIZE = 100
FILTERS = (("", "All"), ("unlabelled", "Unlabelled"), ("true", "True wake"), ("false", "False wake"),
           (dataset.LIKELY_TRUE, "Likely true"), (dataset.LIKELY_FALSE, "Likely false"),
           (dataset.NEAR_MISS, "Near miss"), (dataset.UNKNOWN, "Unknown"))
AUTO_NAMES = {dataset.LIKELY_TRUE: "Likely true", dataset.LIKELY_FALSE: "Likely false",
              dataset.NEAR_MISS: "Near miss", dataset.UNKNOWN: "Unknown"}


def _page(request):
    try:
        return max(1, int(request.GET.get("page") or 1))
    except ValueError:
        return 1


@require_GET
def dataset_page(request):
    _admin_only(request)
    label = request.GET.get("label", "")
    if label not in dict(FILTERS):
        label = ""
    page = _page(request)
    tz = timezone(LODGE_UTC_OFFSET, "PT")  # the lodge's clock, like the assistant's
    cfg = config.current()
    items = []
    for row in dataset.rows(label, PAGE_SIZE, (page - 1) * PAGE_SIZE):
        try:
            row["at_local"] = _fmt(datetime.fromisoformat(row["at"]).astimezone(tz))
        except (TypeError, ValueError):
            row["at_local"] = row.get("at") or ""
        row["auto_name"] = AUTO_NAMES.get(row["auto_label"], row["auto_label"] or "")
        row["word_label"] = WAKE_WORDS.get(row["word"], ("", row["word"] or ""))[1]
        items.append(row)
    total = dataset.count(label)
    return render(request, "voice/settings_dataset.html", {
        "active": "voice", "items": items, "label": label, "filters": FILTERS,
        "counts": dataset.counts(), "total": total, "page": page,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page * PAGE_SIZE < total else None,
        "cfg": cfg, "timezone": "Pacific (UTC-7)",
        "pct": {"collect": round(cfg.collect_threshold * 100), "wake": round(cfg.threshold * 100)},
    })


@require_GET
def dataset_clip(request, clip_id):
    _admin_only(request)
    path = dataset.wav_path(clip_id)
    if path is None or not path.is_file():
        raise Http404
    return FileResponse(path.open("rb"), content_type="audio/wav", filename=f"{clip_id}.wav")


@require_POST
def dataset_label(request, clip_id):
    _admin_only(request)
    value = request.POST.get("label", "")
    label = {"true": "true", "false": "false", "": None, "unset": None}.get(value, "bad")
    if label == "bad" or not dataset.set_label(clip_id, label):
        raise Http404
    back = {k: v for k, v in (("label", request.POST.get("filter", "")),
                              ("page", request.POST.get("page", ""))) if v}
    url = reverse("voice_dataset") + ("?" + urlencode(back) if back else "")
    return HttpResponseRedirect(f"{url}#c-{clip_id}")


@require_GET
def dataset_zip(request):
    _admin_only(request)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    response = StreamingHttpResponse(dataset.zip_stream(), content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="wake-word-clips-{stamp}.zip"'
    return response
