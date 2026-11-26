from django.urls import path

from . import views

urlpatterns = [
    path("", views.status_page, name="voice_status"),
    path("status.json", views.status_json, name="voice_status_json"),
    path("clips/<str:name>", views.clip, name="voice_clip"),
]
