# bdh-app-voice

A [Black Diamond Hub](https://github.com/blackdiamondhub) app that listens for
a wake word ("Hey Jarvis") on the lodge's wall screens. After the wake word it
holds a spoken conversation with Azure OpenAI Realtime. The assistant can act
through the hub's tools gateway, and the screen plays its replies.

The UniFi Connect kiosks stream their microphones to the hub. The app runs
[openWakeWord](https://github.com/dscripka/openWakeWord)'s models with plain
`onnxruntime` and `numpy`, without the `openwakeword` package and its
scipy/scikit-learn/tflite dependencies. Its scores match `openwakeword` 0.6.0
exactly.

## The listen socket

```
ws://<hub>:<voice app port>/voice/api/listen?display=<screen LAN ip>&v=<kiosk version>
```

The hub gives each container app its own port (9100 and up). It proxies the
socket to the app and passes the screen's address in `X-Forwarded-For`. A
kiosk can find the port with the hub's `GET /display/api/kiosk/`, which returns
`{"apps": {"voice": "http://<host>:<port>"}}`.

There is no sign-in. Only private (LAN) addresses are accepted; anything else
is closed with code `4403`.

**Screen → app.** The screen sends binary frames of 16 kHz signed 16-bit
little-endian mono PCM, normally 20 ms (640 bytes) each. Any frame size works.

**App → screen.** The app sends text JSON:

| Message | When |
| --- | --- |
| `{"type":"hello","words":["hey_jarvis"],"threshold":0.5,"assistant":true}` | Right after the socket is accepted. It is sent again if the wake word or the assistant setting changes. |
| `{"type":"wake","word":"hey_jarvis","score":0.91,"at":"2026-…Z"}` | On each detection. A connection gets at most one wake per 2 s of audio. |
| `{"type":"config","listen":false}` | When listening is off. The socket is then closed; the screen should retry later. |

Each connection has its own detector. Inference runs in a small thread pool.

### Conversations

After a wake the same socket carries the conversation. Wake detection is
paused on that screen until the conversation ends. The screen keeps sending
its mic audio, which the assistant needs to hear interruptions.

**App → screen** adds these messages:

| Message | Meaning |
| --- | --- |
| `{"type":"state","state":"listening"}` | The conversation state: `listening`, `thinking`, `speaking` or `idle`. Use it for an indicator. |
| `{"type":"transcript","role":"user","text":"…","final":false}` | A caption. `role` is `user` or `assistant`. Partial captions (`final:false`) grow, and a `final:true` caption replaces them. |
| `{"type":"speak_start"}` | Reply audio follows. |
| binary frames | Reply audio: 24 kHz signed 16-bit little-endian mono PCM, at most 9600 bytes (200 ms) per frame. |
| `{"type":"speak_end"}` | The reply's audio is complete. Play out what is queued. |
| `{"type":"speak_stop"}` | The user interrupted. Drop all queued audio now. |
| `{"type":"conversation_end","reason":"idle"}` | The conversation is over, and wake detection resumes. `reason` is one of `idle`, `max_seconds`, `error`, `ended`, `disabled`, `stopped`, `not_configured` or `busy`. `message` may carry a short text to show. |

**Screen → app** adds one message: `{"type":"end"}` ends the conversation, for
example when the doorbell talk-back takes over the mic.

A screen that ignores the new messages still gets `wake` and works as before.

## The assistant

```
screen mic ─16 kHz─▶ voice app ─24 kHz base64─▶ Azure OpenAI Realtime (GA, wss)
screen speaker ◀─24 kHz PCM── voice app ◀─audio deltas── │
                              │  function calls          │
                              └─▶ hub tools gateway ─────┘ (results)
```

On a wake, the app:

1. opens a tools session with the gateway (`tools.open`, kind `voice`, display
   = the screen's IP). The hub picks the mode: guest during a stay, dashboard
   otherwise, never owner.
2. lists the tools and maps each one to a Realtime function tool.
3. connects to `wss://<endpoint>/openai/v1/realtime?model=<model>` with the
   `api-key` header and sends `session.update`. That update sets the
   instructions (the lodge, the local time at a fixed UTC−7, short replies, and
   no claims without a tool result), the tools, `semantic_vad` turn detection,
   `far_field` noise reduction, `whisper-1` transcription and the voice.
4. streams the mic audio, upsampled from 16 to 24 kHz. Audio heard after the
   wake word and while connecting is buffered and sent first.

The model ends turns by itself. Tool calls run through `tools.call` in a
worker thread, and each result is sent back as a `function_call_output`,
followed by `response.create`. An expired tools session is reopened once. A
refusal is passed to the model as an error, so it can say it can't do that
here.

When the user speaks over a reply, the app:

- sends `speak_stop` to the screen;
- truncates the reply at the point the screen has played;
- cancels the response if it is still running.

A conversation ends:

- after **Wait for a follow-up** seconds of silence once the last reply has
  played;
- after **Longest conversation** seconds in total;
- on a Realtime or socket error;
- when the screen sends `end` or disconnects.

At most 2 conversations run at once. A third wake gets
`conversation_end` with reason `busy`.

Each turn is published on the app's own event topic `voice/conversation`. A
turn record holds:

```json
{"screen": "192.168.10.50", "ip": "192.168.10.50", "sid": "…", "mode": "guest",
 "model": "gpt-realtime-2.1-mini", "turn": 1, "at": "…Z",
 "user": "turn on the fireplace", "assistant": "The fireplace is on.",
 "tools": [{"name": "fireplace_on", "args": {}, "ok": true}]}
```

The hub audits every tool call on its own `assistant/<sid>/…` events.

### Setting it up

1. In Settings → Voice, paste the key of the Azure OpenAI resource into
   **Azure OpenAI API key** (the `api_key` secret). The key is stored in the
   hub's secret store and never shown again.
2. Leave **Azure OpenAI endpoint** at `https://bdh-ai-lodge.openai.azure.com`,
   or point it at another resource that has the realtime deployments.
3. Make sure **Voice assistant** is on.

Without a key the wake still chimes. The screen then gets `conversation_end`
with reason `not_configured` and the message "The assistant isn't set up yet."

### Network

The app uses `lan` (host) networking. Screens connect to it on the LAN, and
the same mode reaches Azure over the internet directly. The WebSocket client
also honours `HTTPS_PROXY` if the hub ever moves the app behind its outbound
proxy.

### Cost and latency

`gpt-realtime-2.1-mini` is billed per audio and text token. A short question
and answer uses about 500–1,000 tokens. The instructions and tool list are
also sent with each response, and text input tokens cost far less than audio
tokens. Every conversation also pays for `whisper-1` transcription of the
user's speech. The cost is a few cents per conversation at most. The full
`gpt-realtime-2.1` model costs several times more.

In a live test, the first reply audio arrived about 1.8 s after the user
stopped speaking. Replies after a tool call took about 0.5 s more.

## Settings

Admins can change these settings:

- **Listen for the wake word**: on by default.
- **Wake word**: Hey Jarvis (default), Alexa or Hey Mycroft.
- **Detection threshold**: in percent, 50 by default. The hello message sends it as 0..1.
- **Save detection clips**: on by default.
- **Voice assistant** (`enabled`): on by default. Off makes the screen only chime.
- **Azure OpenAI API key** (`api_key`, secret).
- **Azure OpenAI endpoint** (`endpoint`): `https://bdh-ai-lodge.openai.azure.com` by default.
- **Model** (`model`): `gpt-realtime-2.1-mini` (default) or `gpt-realtime-2.1`.
- **Voice** (`voice`): `marin` (default), `cedar`, `alloy`, `ash`, `ballad`,
  `coral`, `echo`, `sage`, `shimmer` or `verse`.
- **Longest conversation** (`max_seconds`): 90 s by default, from 10 to 600 s.
- **Wait for a follow-up** (`follow_up_seconds`): 8 s by default, from 2 to 60 s.
- **Extra instructions** (`instructions`): text added to the assistant's
  instructions, such as house rules or local tips.

Settings → Voice is for admins only. It shows:

- the connected screens, with each screen's live mic level, its peak score over the last 10 s, and whether a conversation is running;
- the last 50 detections, with each clip's audio.

Clips cover 2 s before and 1 s after the wake. The app keeps the newest 100
under `/data/voice/clips`, outside the media folder, so they are only served
to admins.

## Development

```
pip install -e .
python manage.py test voice   # from a hub checkout with this app installed
```

The tests include two 16 kHz clips made with Windows speech synthesis:

- `voice/tests/fixtures/hey_jarvis.wav` scores 0.997.
- `voice/tests/fixtures/other_speech.wav` scores 0.004.
- `voice/tests/fixtures/question.wav` says "What time is it?". The live test
  below uses it.

`voice/tests/test_assistant.py` runs whole conversations against a fake
Realtime server and the SDK's `FakeBroker`. One live test talks to Azure. It
is skipped unless `VOICE_LIVE_TEST=1` and `BDH_VOICE_API_KEY` are set.
`VOICE_LIVE_ENDPOINT` and `VOICE_LIVE_MODEL` are optional.

## Licenses

The app's code is licensed under [Apache-2.0](LICENSE).

The ONNX models in `voice/models/` come from openWakeWord's
[v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1):

- `melspectrogram.onnx` and `embedding_model.onnx` are Apache-2.0.
- The pre-trained wake word models (`hey_jarvis`, `alexa` and `hey_mycroft`)
  are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  They are for non-commercial use only. This private hub is non-commercial.
