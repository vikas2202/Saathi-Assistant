"""Hidden real-Tk checks with synthetic frames. No camera, microphone or API calls."""
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from saathi.config import Config
from saathi.ui import App
from saathi.vision import Snapshot, Face
from saathi.recognition import Enrollment


def main():
    with tempfile.TemporaryDirectory() as temp, patch("saathi.ui.ROOT", Path(temp)):
        root = tk.Tk()
        root.withdraw()
        app = App(root, Config(stable_frames=2))
        app.voice_enabled.set(False)
        root.update_idletasks()
        v = np.zeros(128, np.float32)
        v[0] = 1
        person = app.store.enroll("Saiyam", [v]*5, True)
        app.refresh_profiles()
        def shot(faces):
            return Snapshot(np.zeros((480, 640, 3), np.uint8), faces, time.monotonic(), 12, 20, {})
        face = Face((100, 100, 120, 120), v)
        app.handle_frame(shot([face]))
        app.handle_frame(shot([face]))
        assert app.gate.active == person
        assert "Saiyam" in app.identity.get()
        assert "Hi Saiyam" in app.transcript.get("1.0", "end")
        previous = app.token()
        app.history.append({"role": "user", "content": "private study note"})
        app.handle_frame(shot([face, face]))
        assert app.gate.active is None
        assert not app.current(previous)
        assert app.history == []
        assert "private" not in app.transcript.get("1.0", "end")
        app.handle_frame(shot([face]))
        app.handle_frame(shot([face]))
        token = app.token()
        app.events.put(("reply", app.job_id, previous, ("old", "STALE PRIVATE REPLY")))
        app.tick()
        assert "STALE PRIVATE" not in app.transcript.get("1.0", "end")
        app.enrollment = Enrollment("Other", True)
        app.enrollment_frame(shot([]))
        assert app.enrollment is None
        assert len(app.store.profiles()) == 1
        app.stop_camera()
        assert app.snapshot is None and app.gate.active is None
        app.close()
        print("PASS: real Tk layout, recognition greeting, multi-person isolation, stale reply rejection, enrollment cancellation, and stop cleanup.")


if __name__ == "__main__":
    main()
