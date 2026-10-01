import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock
import wave

import httpx
import numpy as np
from openai import OpenAI

from saathi.audio import record_audio
from saathi.config import Config
from saathi.conversation import CloudAssistant
from saathi.vision import CameraWorker, quality_reason


class CloudSDKTests(unittest.TestCase):
    def test_actual_sdk_serialization_and_response_parsing_without_network(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "id": "resp_test", "object": "response", "created_at": 1,
                "model": "gpt-4.1-mini", "status": "completed",
                "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text", "text": "Hello Saiyam.", "annotations": []}]}]
            })
        transport = httpx.MockTransport(handler)
        with OpenAI(api_key="test-key-not-real", http_client=httpx.Client(transport=transport)) as client:
            result = CloudAssistant("", Config(), client).reply(None, [], "Hello")
        self.assertEqual(result, "Hello Saiyam.")
        body = json.loads(requests[0].content)
        self.assertFalse(body["store"])
        self.assertEqual(body["input"], [{"role": "user", "content": "Hello"}])

    def test_transcription_multipart_and_empty_error(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"text": "My name is Saiyam"})
        with OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            text = CloudAssistant("", Config(), client).transcribe(b"test-audio")
        self.assertEqual(text, "My name is Saiyam")
        self.assertIn(b"test-audio", requests[0].content)
        self.assertIn("multipart/form-data", requests[0].headers["content-type"])

    def test_authentication_failure_sanitized(self):
        def handler(request):
            return httpx.Response(401, json={"error": {"message": "SECRET", "type": "invalid_request_error", "code": "invalid_api_key"}})
        with OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            with self.assertRaisesRegex(RuntimeError, "key was rejected"):
                CloudAssistant("", Config(), client).reply(None, [], "hi")


class CameraTests(unittest.TestCase):
    @patch("saathi.vision.FaceEngine")
    @patch("cv2.VideoCapture")
    def test_camera_open_failure_releases_both_backends(self, capture, engine):
        camera = Mock()
        camera.isOpened.return_value = False
        capture.return_value = camera
        worker = CameraWorker(Config())
        worker._run()
        self.assertEqual(camera.release.call_count, 2)
        events = list(worker.events.queue)
        self.assertTrue(any(kind == "error" and "Cannot open" in text for kind, text in events))

    @patch("saathi.vision.FaceEngine")
    @patch("cv2.VideoCapture")
    def test_disconnect_invalidates_and_releases(self, capture, engine):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        capture.return_value = camera
        worker = CameraWorker(Config())
        worker.stop_event.wait = Mock()
        worker._run()
        events = list(worker.events.queue)
        self.assertEqual(sum(kind == "invalidate" for kind, _ in events), 5)
        self.assertTrue(any(kind == "error" for kind, _ in events))
        camera.release.assert_called_once()

    def test_bad_lighting_blur_and_cropped_faces(self):
        config = Config()
        self.assertIn("closer", quality_reason(np.zeros((200, 200), np.uint8), (0, 0, 20, 20), config))
        self.assertIn("whole face", quality_reason(np.zeros((200, 200), np.uint8), (-5, 0, 100, 100), config))
        self.assertIn("More light", quality_reason(np.zeros((200, 200), np.uint8), (0, 0, 100, 100), config))
        self.assertIn("bright", quality_reason(np.full((200, 200), 255, np.uint8), (0, 0, 100, 100), config))
        self.assertIn("Hold still", quality_reason(np.full((200, 200), 128, np.uint8), (0, 0, 100, 100), config))


class AudioTests(unittest.TestCase):
    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices", return_value={"default_samplerate": 16000})
    def test_silence_does_not_reach_transcription(self, devices, stream):
        stream.return_value.__enter__.return_value.read.return_value = (np.zeros((1600, 1), np.int16), False)
        with self.assertRaisesRegex(ValueError, "No clear speech"):
            record_audio(2)

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices", return_value={"default_samplerate": 16000})
    def test_recording_is_valid_in_memory_wav(self, devices, stream):
        stream.return_value.__enter__.return_value.read.return_value = (np.full((1600, 1), 2000, np.int16), False)
        audio = record_audio(2)
        with wave.open(io.BytesIO(audio), "rb") as wav:
            self.assertEqual(wav.getnframes(), 32000)
            self.assertEqual(wav.getnchannels(), 1)

    @patch("sounddevice.InputStream")
    @patch("sounddevice.query_devices", return_value={"default_samplerate": 16000})
    def test_cancel_releases_microphone(self, devices, stream):
        import threading
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(ValueError, "cancelled"):
            record_audio(2, cancel=cancel)
        stream.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
