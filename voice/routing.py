"""Websocket routes (matched against the full path, so they carry the ``voice/`` prefix)."""
from django.urls import path

from .listen import listen

LISTEN_PATH = "/voice/api/listen"

websocket_urlpatterns = [
    path(LISTEN_PATH.lstrip("/"), listen, name="voice_listen"),
]
