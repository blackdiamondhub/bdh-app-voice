"""Sample-rate conversion between the kiosk (16 kHz) and the realtime API (24 kHz)."""
import numpy as np

KIOSK_RATE = 16000
MODEL_RATE = 24000


class Resampler:
    """Streaming linear-interpolation resampler for int16 mono audio.

    It keeps the last input sample and the fractional read position between
    calls, so chunks of any size join without clicks and the output length
    tracks ``len(input) * dst / src`` over the whole stream.
    """

    def __init__(self, src=KIOSK_RATE, dst=MODEL_RATE):
        self.step = src / dst
        self.pos = 0.0  # read position in the next buffer (index 0 = carried sample)
        self.prev = None

    def __call__(self, pcm):
        pcm = np.asarray(pcm, dtype=np.int16)
        if not len(pcm):
            return np.zeros(0, dtype=np.int16)
        x = pcm.astype(np.float32)
        if self.prev is not None:
            x = np.concatenate(([self.prev], x))
        last = len(x) - 1
        if self.pos > last:
            self.pos -= last
            self.prev = x[-1]
            return np.zeros(0, dtype=np.int16)
        n = int((last - self.pos) // self.step) + 1
        where = self.pos + np.arange(n) * self.step
        out = np.interp(where, np.arange(len(x)), x)
        self.pos = where[-1] + self.step - last
        self.prev = x[-1]
        return np.clip(np.rint(out), -32768, 32767).astype(np.int16)
