from bdh_sdk import PluginSpec, Setting, SettingsPage
from bdh_sdk.plugins import BOOL, CHOICE, INT, LIFECYCLE_ALWAYS, NETWORK_LAN

spec = PluginSpec(
    id="voice",
    app="voice",
    urls="voice.urls",
    # /voice/api/listen: the wall screens stream their microphones here.
    websocket_routes="voice.routing.websocket_urlpatterns",
    settings=(
        Setting("listen", "Listen for the wake word", BOOL, True,
                help="Off: screens are told to stop streaming and retry later."),
        Setting("wake_word", "Wake word", CHOICE, "hey_jarvis",
                choices=(("hey_jarvis", "Hey Jarvis"), ("alexa", "Alexa"),
                         ("hey_mycroft", "Hey Mycroft"))),
        Setting("threshold", "Detection threshold", INT, 50, minimum=1, maximum=99, unit="%",
                help="Score a wake word needs. Lower hears more, and falsely more."),
        Setting("save_clips", "Save detection clips", BOOL, True,
                help="Keep 3 s of audio around each detection (newest 100) for admins."),
    ),
    # Admins only: connected screens, live levels and recent detections.
    settings_pages=(
        SettingsPage("Voice", "voice_status", subtitle="Wake word screens and detections",
                     icon="fas fa-microphone"),
    ),
    api_version=2,
    # The kiosks reach the app through the hub on the LAN.
    network=NETWORK_LAN,
    # Screens keep their listen socket open all the time.
    lifecycle=LIFECYCLE_ALWAYS,
    memory_mb=256,
)
