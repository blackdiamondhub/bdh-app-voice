from bdh_sdk import PluginSpec, Setting, SettingsPage
from bdh_sdk.plugins import BOOL, CHOICE, INT, LIFECYCLE_ALWAYS, NETWORK_LAN, SECRET, STR

from .config import (DEFAULT_ENDPOINT, DEFAULT_FOLLOW_UP, DEFAULT_MAX_SECONDS, DEFAULT_MODEL,
                     DEFAULT_VOICE, MODELS, VOICES)

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
        Setting("enabled", "Voice assistant", BOOL, True,
                help="Talk with the assistant after the wake word. Off: the screen only chimes."),
        Setting("api_key", "Azure OpenAI API key", SECRET,
                help="Key of the Azure OpenAI resource that hosts the realtime models."),
        Setting("endpoint", "Azure OpenAI endpoint", STR, DEFAULT_ENDPOINT,
                help="https://<resource>.openai.azure.com"),
        Setting("model", "Model", CHOICE, DEFAULT_MODEL, choices=MODELS,
                help="Realtime deployment to talk to."),
        Setting("voice", "Voice", CHOICE, DEFAULT_VOICE, choices=VOICES),
        Setting("max_seconds", "Longest conversation", INT, DEFAULT_MAX_SECONDS,
                minimum=10, maximum=600, unit="s"),
        Setting("follow_up_seconds", "Wait for a follow-up", INT, DEFAULT_FOLLOW_UP,
                minimum=2, maximum=60, unit="s",
                help="How long to keep listening after a reply before the conversation ends."),
        Setting("instructions", "Extra instructions", STR, "",
                help="Added to the assistant's instructions, e.g. house rules or local tips."),
        Setting("screen_rooms", "Where the screens are", STR, "",
                help="So \"here\" means something, e.g. \"the great room on the main floor, by the "
                     "Living Room Fireplace\". Several screens: \"192.168.10.195=great room; "
                     "192.168.10.196=kitchen\"."),
    ),
    # Admins only: connected screens, live levels and recent detections.
    settings_pages=(
        SettingsPage("Voice", "voice_status", subtitle="Wake word screens and detections",
                     icon="fas fa-microphone"),
    ),
    api_version=2,
    # tools: the assistant acts through the hub's tools gateway (guest or
    # dashboard mode, never owner). Events on voice/# need no permission.
    permissions=("tools",),
    # The kiosks reach the app through the hub; the realtime API is on the
    # internet. Host networking (lan) also reaches it directly; if an admin
    # hasn't allowed lan the app runs as outbound and goes through the proxy.
    network=NETWORK_LAN,
    # Screens keep their listen socket open all the time.
    lifecycle=LIFECYCLE_ALWAYS,
    memory_mb=256,
)
