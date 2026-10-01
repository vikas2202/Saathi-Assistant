from dataclasses import dataclass
import queue
import sys
import threading
import time

import numpy as np

from .config import ROOT
from .model_assets import verify_models
from .recognition import normalize


@dataclass
class Face:
    box: tuple
    embedding: object = None
    quality: str = ""


@dataclass
class Snapshot:
    frame: object
    faces: list
    captured_at: float
    fps: float
    inference_ms: float
    object_counts: dict


def quality_reason(gray, box, config):
    import cv2
    x, y, w, h = map(int, box)
    height, width = gray.shape[:2]
    if min(w, h) < config.min_face_pixels:
        return "Move closer"
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return "Keep your whole face in view"
    crop = gray[y:y + h, x:x + w]
    brightness = float(crop.mean())
    if brightness < config.min_brightness:
        return "More light needed"
    if brightness > config.max_brightness:
        return "Reduce bright light"
    if cv2.Laplacian(crop, cv2.CV_64F).var() < config.min_blur_score:
        return "Hold still / sharpen focus"
    return ""


class FaceEngine:
    def __init__(self, config):
        import cv2
        directory = verify_models()
        self.config = config
        cv2.setNumThreads(2)
        self.detector = cv2.FaceDetectorYN.create(str(directory / "yunet.onnx"), "", (320, 320), 0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(directory / "sface.onnx"), "")

    def analyze(self, frame):
        import cv2
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, detections = self.detector.detect(frame)
        if detections is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        for row in detections:
            box = tuple(int(v) for v in row[:4])
            reason = quality_reason(gray, box, self.config)
            embedding = None
            if not reason and len(detections) <= 4:
                aligned = self.recognizer.alignCrop(frame, row)
                embedding = normalize(self.recognizer.feature(aligned))
            elif len(detections) > 4:
                reason = "Too many faces for a personal conversation"
            faces.append(Face(box, embedding, reason))
        return faces


class CameraWorker:
    def __init__(self, config):
        self.config = config
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        import cv2
        camera = None
        try:
            self.events.put(("status", "Loading face models..."))
            engine = FaceEngine(self.config)
            objects = None
            if self.config.objects_model:
                from pathlib import Path
                path = Path(self.config.objects_model)
                path = path if path.is_absolute() else ROOT / path
                if not path.is_file():
                    raise ValueError("Object model is missing. Set objects_model to an existing local YOLO checkpoint.")
                from ultralytics import YOLO
                objects = YOLO(str(path))
            if self.stop_event.is_set():
                return
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.config.camera_index, backend)
            if not camera.isOpened():
                camera.release()
                camera = cv2.VideoCapture(self.config.camera_index)
            if not camera.isOpened():
                raise ValueError("Cannot open the webcam. Close other camera apps, check camera permissions, or change camera_index.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.events.put(("status", "Camera active. Face the camera in good light."))
            last = time.monotonic()
            failures = 0
            fps = 0
            while not self.stop_event.is_set():
                begin = time.monotonic()
                ok, frame = camera.read()
                if not ok or frame is None:
                    failures += 1
                    self.events.put(("invalidate", "Camera frame unavailable"))
                    if failures >= 5:
                        raise ValueError("The webcam stopped sending frames. Reconnect it and restart the camera.")
                    self.stop_event.wait(0.15)
                    continue
                failures = 0
                captured = time.monotonic()
                # Bound inference input even if a driver ignores requested resolution.
                scale = min(self.config.frame_width / frame.shape[1], self.config.frame_height / frame.shape[0], 1)
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                faces = engine.analyze(frame)
                counts = {}
                if objects is not None:
                    result = objects.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4,
                                           imgsz=320, device="cpu", verbose=False)[0]
                    frame = result.plot()
                    if result.boxes is not None:
                        for value in result.boxes.cls.tolist():
                            label = result.names[int(value)]
                            counts[label] = counts.get(label, 0) + 1
                elapsed = time.monotonic() - captured
                current = time.monotonic()
                rate = 1 / max(current - last, 1e-6)
                fps = rate if not fps else 0.85 * fps + 0.15 * rate
                last = current
                snapshot = Snapshot(frame, faces, captured, fps, elapsed * 1000, counts)
                if self.frames.full():
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                self.frames.put_nowait(snapshot)
                self.stop_event.wait(max(0, 1 / self.config.target_fps - (time.monotonic() - begin)))
        except Exception as exc:
            # Camera/model exceptions contain no user credentials.
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            if camera is not None:
                camera.release()
            self.events.put(("stopped", "Camera stopped"))
