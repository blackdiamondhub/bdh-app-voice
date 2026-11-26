# bdh-app-voice

A [Black Diamond Hub](https://github.com/blackdiamondhub) app that listens for
a wake word ("Hey Jarvis") on the lodge's wall screens. For now it only
detects the word and tells the screen. Later it will grow into a voice assistant.

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
| `{"type":"hello","words":["hey_jarvis"],"threshold":0.5}` | Right after the socket is accepted, and again if the wake word is changed. |
| `{"type":"wake","word":"hey_jarvis","score":0.91,"at":"2026-…Z"}` | On each detection. A connection gets at most one wake per 2 s of audio. |
| `{"type":"config","listen":false}` | When listening is off. The socket is then closed; the screen should retry later. |

Each connection has its own detector. Inference runs in a small thread pool.

## Settings

Admins can change these settings:

- **Listen for the wake word**: on by default.
- **Wake word**: Hey Jarvis (default), Alexa or Hey Mycroft.
- **Detection threshold**: in percent, 50 by default. The hello message sends it as 0..1.
- **Save detection clips**: on by default.

Settings → Voice is for admins only. It shows:

- the connected screens, with each screen's live mic level and its peak score over the last 10 s;
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

## Licenses

The app's code is licensed under [Apache-2.0](LICENSE).

The ONNX models in `voice/models/` come from openWakeWord's
[v0.5.1 release](https://github.com/dscripka/openWakeWord/releases/tag/v0.5.1):

- `melspectrogram.onnx` and `embedding_model.onnx` are Apache-2.0.
- The pre-trained wake word models (`hey_jarvis`, `alexa` and `hey_mycroft`)
  are [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  They are for non-commercial use only. This private hub is non-commercial.
