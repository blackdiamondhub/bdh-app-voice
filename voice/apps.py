import sys

from django.apps import AppConfig


def in_container(argv=None):
    """``python -m bdh_sdk.runtime`` (the app's container) runs ``bdh_sdk/runtime/__main__.py``."""
    argv = sys.argv if argv is None else argv
    parts = (argv[0] if argv else "").replace("\\", "/").split("/")
    return parts[-3:] == ["bdh_sdk", "runtime", "__main__.py"]


class VoiceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "voice"
    verbose_name = "Voice"

    def ready(self):
        if in_container():
            from . import conversations

            # Close records left open by a restart, prune old ones, answer the sentinel.
            conversations.start()
