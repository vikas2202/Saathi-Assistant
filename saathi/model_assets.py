"""Download only known OpenCV model assets and verify their published LFS hashes."""
import hashlib
from pathlib import Path
import urllib.request

from .config import ROOT

ASSETS = {
    "yunet.onnx": (
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", 232589),
    "sface.onnx": (
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", 38696353),
}


def valid(path, expected_hash, expected_size):
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == expected_hash


def download_models(directory=None):
    directory = Path(directory) if directory else ROOT / "weights"
    directory.mkdir(parents=True, exist_ok=True)
    for name, (source, digest, size) in ASSETS.items():
        target = directory / name
        if valid(target, digest, size):
            print(f"Verified {name}")
            continue
        print(f"Downloading {name} ({size / 1e6:.1f} MB)...", flush=True)
        url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/" + source
        temp = target.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=45) as response, temp.open("wb") as handle:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError(f"Unexpected file size for {name}")
                    handle.write(chunk)
            if not valid(temp, digest, size):
                raise ValueError(f"Integrity check failed for {name}; no model was installed")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        print(f"Verified {name}")


def verify_models(directory=None):
    directory = Path(directory) if directory else ROOT / "weights"
    for name, (_, digest, size) in ASSETS.items():
        if not valid(directory / name, digest, size):
            raise ValueError("Models are missing or damaged. Run: python -m saathi download-models")
    return directory
