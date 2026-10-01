import argparse
import importlib
import json
from pathlib import Path
import sys

from .config import Config, ROOT


def doctor():
    errors = 0
    print(f"Python: {sys.version.split()[0]}")
    for name in ("numpy", "cv2", "PIL", "openai", "sounddevice", "pyttsx3", "tkinter"):
        try:
            module = importlib.import_module(name)
            print(f"OK {name}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:
            print(f"MISSING {name}: {type(exc).__name__}")
            errors += 1
    try:
        from .model_assets import verify_models
        verify_models()
        print("OK face model hashes")
    except ValueError as exc:
        print(str(exc))
        errors += 1
    print("Camera, microphone, speakers and API connectivity require an interactive check.")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description="Saathi face recognition and voice assistant")
    parser.add_argument("--config", type=Path, help="JSON settings file")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Open the desktop app (default)")
    sub.add_parser("download-models", help="Download and checksum the two face models")
    sub.add_parser("doctor", help="Check dependencies and model integrity without opening devices")
    sub.add_parser("audio-devices", help="List microphone/speaker indexes")
    sub.add_parser("smoke-models", help="Run real models on a synthetic blank image; no webcam")
    cap = sub.add_parser("capture-eval", help="Capture labeled embeddings for calibration")
    label = cap.add_mutually_exclusive_group(required=True)
    label.add_argument("--name", help="Exact name of an enrolled participant")
    label.add_argument("--unknown", action="store_true", help="Participant has no enrolled profile")
    cap.add_argument("--split", required=True, choices=("tune", "test"))
    cap.add_argument("--count", type=int, default=15)
    cap.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation.jsonl")
    cap.add_argument("--consent", action="store_true", help="Participant agreed to save these face embeddings")
    cal = sub.add_parser("calibrate", help="Tune thresholds and evaluate a separate held-out split")
    cal.add_argument("--data", type=Path, default=ROOT / "data" / "evaluation.jsonl")
    cal.add_argument("--report", type=Path, default=ROOT / "calibration-report.json")
    cal.add_argument("--max-far", type=float, default=0.01)
    cal.add_argument("--apply", action="store_true", help="Apply only if the held-out false-match check passes")
    args = parser.parse_args()
    try:
        config = Config.load(args.config)
        if args.command == "download-models":
            from .model_assets import download_models
            download_models()
        elif args.command == "doctor":
            return doctor()
        elif args.command == "audio-devices":
            import sounddevice
            print(sounddevice.query_devices())
        elif args.command == "smoke-models":
            import numpy as np
            from .vision import FaceEngine
            engine = FaceEngine(config)
            assert engine.analyze(np.zeros((480, 640, 3), np.uint8)) == []
            vector = engine.recognizer.feature(np.zeros((112, 112, 3), np.uint8))
            assert vector.size == 128 and np.isfinite(vector).all()
            print("PASS: YuNet blank-frame rejection and SFace 128-value inference. This does not measure face-recognition accuracy.")
        elif args.command == "capture-eval":
            from .calibration import capture
            capture(config, args.name, args.unknown, args.split, args.count, args.output, args.consent)
        elif args.command == "calibrate":
            from .calibration import calibrate, load_rows, apply_report
            from .storage import Store
            templates = Store(ROOT / "data" / "profiles.sqlite3").templates()
            report = calibrate(load_rows(args.data), templates, args.max_far)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))
            if args.apply:
                apply_report(report, templates, args.config or ROOT / "config.local.json")
                print("Applied calibrated settings. Restart the app.")
        else:
            from .ui import launch
            launch(config)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, ImportError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
