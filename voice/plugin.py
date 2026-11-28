from bdh_sdk import PluginSpec, Setting, SettingsPage
from bdh_sdk.plugins import BOOL, CHOICE, INT, LIFECYCLE_ALWAYS, NETWORK_LAN, SECRET, STR

from .config import (DEFAULT_COLLECT_THRESHOLD, DEFAULT_ENDPOINT, DEFAULT_FOLLOW_UP,
                     DEFAULT_MAX_CLIPS, DEFAULT_MAX_SECONDS, DEFAULT_MODEL,
                     DEFAULT_TRANSCRIPT_DAYS, DEFAULT_TRANSCRIPT_GUEST_DAYS, DEFAULT_VOICE,
                     MODELS, VOICES)
from .detector import WAKE_WORDS

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
                choices=tuple((word, label) for word, (_, label) in WAKE_WORDS.items())),
        Setting("threshold", "Detection threshold", INT, 50, minimum=1, maximum=99, unit="%",
                help="Score a wake word needs. Lower hears more, and falsely more."),
        Setting("save_clips", "Save detection clips", BOOL, True,
                help="Keep 3 s of audio around each detection (newest 100) for admins. "
                     "Never saved while guests are staying, only between stays and during "
                     "owner stays."),
        Setting("collect_clips", "Collect wake-word training clips", BOOL, True,
                help="Stores 3 s of microphone audio ON THE HUB around every wake and near-miss "
                     "(scores at or above the collect threshold), with what was said after a "
                     "wake, so a custom wake word can be trained on real audio. Never sent "
                     "anywhere; only admins can listen, label or download them (Settings → "
                     "Wake-word clips)."),
        Setting("collect_threshold", "Collect threshold", INT, DEFAULT_COLLECT_THRESHOLD,
                minimum=1, maximum=99, unit="%",
                help="Score that stores a 3 s audio clip on the hub. Below the detection "
                     "threshold, so near-misses are kept too."),
        Setting("collect_during_stays", "Collect clips during guest stays", BOOL, False,
                help="Audio around wake words and near-misses is stored on the hub. Off: none "
                     "is stored while guests are staying (for their privacy), only between "
                     "stays and during owner stays."),
        Setting("collect_max_clips", "Clips to keep", INT, DEFAULT_MAX_CLIPS,
                minimum=10, maximum=50000,
                help="Wake-word clips stored on the hub (about 100 KB each). The oldest "
                     "unlabelled clips are deleted first; labelled ones are kept longest."),
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
        Setting("transcript_guest_days", "Keep guest conversations", INT,
                DEFAULT_TRANSCRIPT_GUEST_DAYS, minimum=0, maximum=365, unit="days",
                help="Transcripts of conversations during a guest stay are kept on the hub, for "
                     "admins only, until this many days after the guests check out."),
        Setting("transcript_days", "Keep other conversations", INT, DEFAULT_TRANSCRIPT_DAYS,
                minimum=1, maximum=365, unit="days",
                help="Transcripts of conversations during owner stays and between stays, kept "
                     "this many days for admins only."),
    ),
    # Admins only: connected screens, live levels and recent detections.
    settings_pages=(
        SettingsPage("Voice", "voice_status", subtitle="Wake word screens and detections",
                     icon="fas fa-microphone"),
        SettingsPage("Wake-word clips", "voice_dataset",
                     subtitle="Review, label and download training clips",
                     icon="fas fa-wave-square"),
        SettingsPage("Conversations", "voice_conversations",
                     subtitle="What was said to the assistant and what it did",
                     icon="fas fa-comments"),
    ),
    api_version=2,
    # tools: the assistant acts through the hub's tools gateway (guest or
    # dashboard mode, never owner). Events on voice/# need no permission.
    # stays.read: no training clips are collected while guests stay (by default), and
    # transcripts stay out of the log then.
    # events.subscribe:sentinel/sync: the sentinel asks for conversation records it missed.
    permissions=("tools", "stays.read", "events.subscribe:sentinel/sync"),
    # The kiosks reach the app through the hub; the realtime API is on the
    # internet. Host networking (lan) also reaches it directly; if an admin
    # hasn't allowed lan the app runs as outbound and goes through the proxy.
    network=NETWORK_LAN,
    # Screens keep their listen socket open all the time.
    lifecycle=LIFECYCLE_ALWAYS,
    memory_mb=256,
)
