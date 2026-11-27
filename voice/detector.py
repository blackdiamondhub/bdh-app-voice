"""openWakeWord's streaming inference, with onnxruntime and numpy only.

Mirrors ``openwakeword.utils.AudioFeatures`` (streaming part) and
``openwakeword.model.Model.predict`` for 80 ms (1280-sample) chunks, so the
scores match the ``openwakeword`` package without its scipy/scikit-learn/tflite
dependencies:

1. ``melspectrogram.onnx`` over the newest 1280 samples plus the 480 before
   them (int16 values as float32), transformed with ``x / 10 + 2``, appended to
   a mel buffer that starts as ``ones((76, 32))``;
2. ``embedding_model.onnx`` over the newest 76 mel frames: one 96-dim
   embedding per chunk, appended to a feature buffer that starts with the
   embeddings of 4 s of random noise;
3. the wake model over the newest 16 embeddings: a score 0..1. The first 5
   scores are forced to 0 while the buffers fill, as openWakeWord does.

The ONNX sessions are loaded once per process and shared; each
:class:`Detector` (one per screen) keeps only its own buffers.
"""
import logging
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent / "models"
SAMPLE_RATE = 16000
CHUNK = 1280  # 80 ms: one embedding and one score per chunk
MEL_CONTEXT = 160 * 3  # the extra samples the mel model needs before a chunk
MEL_WINDOW = 76
MEL_BINS = 32
FEATURES = 16  # embeddings a wake model looks at (~1.3 s)
WARMUP_SCORES = 5

# Bundled wake models: setting value -> (file, label).
WAKE_WORDS = {
    "hey_jarvis": ("hey_jarvis_v0.1.onnx", "Hey Jarvis"),
    "alexa": ("alexa_v0.1.onnx", "Alexa"),
    "hey_mycroft": ("hey_mycroft_v0.1.onnx", "Hey Mycroft"),
    # Custom model from blackdiamondhub/bdh-wakeword (release hey_juniper-v2).
    "hey_juniper": ("hey_juniper_v2.onnx", "Hey Juniper"),
}
DEFAULT_WORD = "hey_jarvis"


def _session(name):
    import onnxruntime as ort

    # Hide onnxruntime's start-up warnings (e.g. GPU discovery in containers).
    ort.set_default_logger_severity(3)
    opts = ort.SessionOptions()
    # Many screens share the hub's CPU: one thread per inference keeps each
    # call cheap and lets the executor spread screens over cores.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODELS_DIR / name), sess_options=opts,
                                providers=["CPUExecutionProvider"])


class _Shared:
    """The ONNX sessions every detector uses (``InferenceSession.run`` is thread-safe)."""

    def __init__(self):
        self.mel = _session("melspectrogram.onnx")
        self.embedding = _session("embedding_model.onnx")
        self._wake = {}
        self._lock = threading.Lock()
        rng = np.random.default_rng(0)
        noise = rng.integers(-1000, 1000, SAMPLE_RATE * 4).astype(np.int16)
        self.initial_features = self._embeddings(noise)

    def melspectrogram(self, samples):
        x = np.asarray(samples, dtype=np.float32)[None, :]
        spec = np.squeeze(self.mel.run(None, {"input": x})[0])
        return spec / 10 + 2

    def embed(self, windows):
        """Embeddings of mel windows shaped ``(n, 76, 32)``: ``(n, 96)``."""
        batch = np.asarray(windows, dtype=np.float32)[..., None]
        out = self.embedding.run(None, {"input_1": batch})[0]
        return out.reshape(len(batch), -1)

    def _embeddings(self, audio):
        spec = self.melspectrogram(audio)
        windows = [spec[i:i + MEL_WINDOW] for i in range(0, spec.shape[0], 8)
                   if spec[i:i + MEL_WINDOW].shape[0] == MEL_WINDOW]
        return self.embed(np.array(windows))

    def wake(self, word):
        with self._lock:
            sess = self._wake.get(word)
            if sess is None:
                filename = WAKE_WORDS[word][0]
                sess = self._wake[word] = _session(filename)
                log.info("loaded wake model %s", filename)
            return sess


@lru_cache(maxsize=1)
def shared():
    return _Shared()


class Detector:
    """One audio stream's wake-word state. Not thread-safe: feed it from one thread at a time."""

    def __init__(self, word=DEFAULT_WORD, models=None):
        if word not in WAKE_WORDS:
            raise ValueError(f"unknown wake word {word!r}")
        self.word = word
        self.models = models or shared()
        self.wake_model = self.models.wake(word)
        self._wake_input = self.wake_model.get_inputs()[0].name
        self._n_features = int(self.wake_model.get_inputs()[0].shape[1] or FEATURES)
        self._pending = np.empty(0, dtype=np.int16)
        self._context = np.empty(0, dtype=np.int16)
        self._mel = np.ones((MEL_WINDOW, MEL_BINS), dtype=np.float32)
        self._features = self.models.initial_features[-self._n_features:].copy()
        self.scores = 0

    @property
    def buffered(self):
        """Samples received but not yet scored (less than one chunk)."""
        return len(self._pending)

    def feed(self, pcm):
        """Add int16 samples (any length); return the scores of the 80 ms chunks they completed."""
        pcm = np.asarray(pcm, dtype=np.int16)
        data = np.concatenate((self._pending, pcm)) if self._pending.size else pcm
        n = (len(data) // CHUNK) * CHUNK
        self._pending = data[n:].copy()
        return [self._chunk(data[i:i + CHUNK]) for i in range(0, n, CHUNK)]

    def _chunk(self, chunk):
        audio = np.concatenate((self._context, chunk))
        self._context = audio[-MEL_CONTEXT:]
        spec = self.models.melspectrogram(audio)
        self._mel = np.vstack((self._mel, spec))[-MEL_WINDOW:]
        emb = self.models.embed(self._mel[None])
        self._features = np.vstack((self._features, emb))[-self._n_features:]
        out = self.wake_model.run(None, {self._wake_input: self._features[None].astype(np.float32)})[0]
        score = float(np.asarray(out).reshape(-1)[0])
        self.scores += 1
        return 0.0 if self.scores <= WARMUP_SCORES else score


def read_wav(path):
    """16 kHz mono 16-bit WAV samples as int16."""
    import wave

    with wave.open(str(path), "rb") as f:
        if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1 or f.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz mono 16-bit")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.int16)


def score_clip(samples, word=DEFAULT_WORD, padding=1.0):
    """Max score over a clip padded with ``padding`` s of silence each side (like ``predict_clip``)."""
    pad = np.zeros(int(SAMPLE_RATE * padding), dtype=np.int16)
    det = Detector(word)
    scores = det.feed(np.concatenate((pad, np.asarray(samples, dtype=np.int16), pad)))
    return max(scores) if scores else 0.0
