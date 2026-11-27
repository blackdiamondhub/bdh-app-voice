from django.urls import path

from . import views

urlpatterns = [
    path("", views.status_page, name="voice_status"),
    path("status.json", views.status_json, name="voice_status_json"),
    path("clips/<str:name>", views.clip, name="voice_clip"),
    path("dataset/", views.dataset_page, name="voice_dataset"),
    path("dataset/download.zip", views.dataset_zip, name="voice_dataset_zip"),
    path("dataset/<str:clip_id>.wav", views.dataset_clip, name="voice_dataset_clip"),
    path("dataset/<str:clip_id>/label", views.dataset_label, name="voice_dataset_label"),
]
