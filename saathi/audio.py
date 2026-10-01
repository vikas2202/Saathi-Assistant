"""Explicit microphone capture and interruptible, local system speech."""
import io
import queue
import threading
import time
import wave

import numpy as np


def record_audio(seconds, device=None, cancel=None):
    import sounddevice as sd
    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])
    pieces = []
    # Blocking reads in a background worker keep memory bounded and are cancellable.
    with sd.InputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                        blocksize=max(1, rate // 10)) as stream:
        for _ in range(seconds * 10):
            if cancel and cancel.is_set():
                raise ValueError("Recording cancelled")
            data, overflow = stream.read(max(1, rate // 10))
            if overflow:
                raise ValueError("Microphone audio overflowed. Close other audio apps and retry.")
            pieces.append(data.copy())
    audio = np.concatenate(pieces).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float32) / 32768))))
    if rms < 0.004:
        raise ValueError("No clear speech was heard. Check your microphone and speak closer.")
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio.astype("<i2").tobytes())
        return buffer.getvalue()


class Speaker:
    def __init__(self):
        self.queue = queue.Queue(maxsize=2)
        self.events = queue.Queue()
        self.busy = threading.Event()
        self.closed = threading.Event()
        self.generation = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True, name="speaker")
        self.thread.start()

    def say(self, text):
        self.cancel()
        self.queue.put_nowait((self.generation, text[:1800]))

    def cancel(self):
        with self.lock:
            self.generation += 1
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def close(self):
        self.cancel()
        self.closed.set()

    def _run(self):
        engine = None
        try:
            while not self.closed.is_set():
                try:
                    generation, text = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if generation != self.generation:
                    continue
                self.busy.set()
                try:
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 170)
                    if generation != self.generation or self.closed.is_set():
                        continue
                    engine.say(text)
                    engine.startLoop(False)
                    try:
                        engine.iterate()
                        while engine.isBusy() and not self.closed.is_set():
                            if generation != self.generation:
                                engine.stop()
                                break
                            engine.iterate()
                            time.sleep(0.02)
                    finally:
                        engine.endLoop()
                except Exception:
                    self.events.put("System speech is unavailable. Replies will still appear as text.")
                    engine = None
                finally:
                    self.busy.clear()
        finally:
            if engine:
                engine.stop()
