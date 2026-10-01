"""Tune decision thresholds on one split and report results on a held-out split."""
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .config import Config, ROOT
from .recognition import Matcher, normalize
from .storage import Store


def evaluate(rows, templates, threshold, margin):
    matcher = Matcher(templates, threshold, margin)
    false_accepts = false_rejects = wrong_identity = correct = unknown_count = known_count = 0
    for row in rows:
        expected = row["expected_id"]
        result = matcher.match(row["embedding"])
        if expected is None:
            unknown_count += 1
            false_accepts += result.person_id is not None
            correct += result.person_id is None
        else:
            known_count += 1
            false_rejects += result.person_id is None
            wrong_identity += result.person_id is not None and result.person_id != expected
            correct += result.person_id == expected
    return {
        "samples": len(rows), "known_samples": known_count, "unknown_samples": unknown_count,
        "false_accepts": false_accepts, "false_rejects": false_rejects,
        "wrong_identity": wrong_identity, "correct": correct,
        "unknown_false_accept_rate": false_accepts / unknown_count if unknown_count else None,
        "known_false_reject_rate": false_rejects / known_count if known_count else None,
        "known_wrong_identity_rate": wrong_identity / known_count if known_count else None,
        "known_recognition_rate": (known_count-false_rejects-wrong_identity) / known_count if known_count else None,
    }


def template_fingerprint(templates):
    digest = hashlib.sha256()
    for person_id, vector in sorted(templates, key=lambda t: (t[0], normalize(t[1]).tobytes())):
        digest.update(person_id.encode())
        digest.update(normalize(vector).tobytes())
    return digest.hexdigest()


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["embedding"] = normalize(row["embedding"])
                if row["split"] not in {"tune", "test"}:
                    raise ValueError("Invalid split")
                if not isinstance(row["session"], str) or not row["session"]:
                    raise ValueError("Missing session")
                if row["expected_id"] is not None and not isinstance(row["expected_id"], str):
                    raise ValueError("Invalid expected_id")
                rows.append(row)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid evaluation row {number}: {exc}") from exc
    return rows


def calibrate(rows, templates, max_far=0.01):
    if not templates:
        raise ValueError("Enroll at least one person before calibration")
    if not 0 <= max_far <= 1:
        raise ValueError("max_far must be between 0 and 1")
    known_ids = {person_id for person_id, _ in templates}
    if any(r["expected_id"] is not None and r["expected_id"] not in known_ids for r in rows):
        raise ValueError("Evaluation data refers to a deleted or missing profile. Capture a new dataset.")
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    for split, values in (("tune", tune), ("test", test)):
        if len(values) < 10 or not any(r["expected_id"] is None for r in values) or not any(r["expected_id"] is not None for r in values):
            raise ValueError(f"{split} needs at least 10 samples, including enrolled people and unknown visitors")
    if {r["session"] for r in tune} & {r["session"] for r in test}:
        raise ValueError("Tune and test must use separate capture sessions")
    seen = set()
    enrolled_vectors = {normalize(v).tobytes() for _, v in templates}
    for row in rows:
        fingerprint = normalize(row["embedding"]).tobytes()
        if fingerprint in seen or fingerprint in enrolled_vectors:
            raise ValueError("Duplicate or enrollment samples found in evaluation data. Capture fresh samples.")
        seen.add(fingerprint)
    best = None
    for threshold in np.arange(0.30, 0.851, 0.025):
        for margin in (0.04, 0.06, 0.08, 0.10, 0.12, 0.16):
            metrics = evaluate(tune, templates, float(threshold), margin)
            if (metrics["unknown_false_accept_rate"] > max_far or metrics["wrong_identity"]
                    or metrics["known_recognition_rate"] < 0.8):
                continue
            # Favor recognition rate within the FAR constraint; conservative tie-break.
            rank = (metrics["correct"], float(threshold), margin)
            if best is None or rank > best[0]:
                best = rank, float(threshold), margin, metrics
    if best is None:
        raise ValueError("No candidate meets the false-match target. Improve enrollment and collect more data.")
    _, threshold, margin, tune_metrics = best
    return {
        "recognition_threshold": round(threshold, 3), "recognition_margin": margin,
        "max_far_target": max_far, "min_known_recognition_target": 0.8, "tune": tune_metrics,
        "held_out_test": evaluate(test, templates, threshold, margin),
        "profile_fingerprint": template_fingerprint(templates),
        "note": "Threshold calibration only, not neural-network fine-tuning. Frame samples within a session are correlated; these rates are not population guarantees.",
    }


def apply_report(report, templates, path):
    if report["profile_fingerprint"] != template_fingerprint(templates):
        raise ValueError("Profiles changed after calibration. Recalibrate before applying.")
    metrics = report["held_out_test"]
    if (metrics["wrong_identity"] or metrics["unknown_false_accept_rate"] > report["max_far_target"]
            or metrics["known_recognition_rate"] < report["min_known_recognition_target"]):
        raise ValueError("Held-out evaluation failed the false-match or recognition target. Settings were not changed.")
    path = Path(path)
    values = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    values.update({key: report[key] for key in ("recognition_threshold", "recognition_margin")})
    Config(**values)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def capture(config, name, unknown, split, count, output, consent):
    if not consent:
        raise ValueError("Capture requires --consent from the participant")
    if not 5 <= count <= 100:
        raise ValueError("Capture count must be between 5 and 100")
    import cv2
    from .vision import CameraWorker
    store = Store(ROOT / "data" / "profiles.sqlite3")
    expected = None
    if not unknown:
        profile = next((p for p in store.profiles() if p.name.casefold() == name.casefold()), None)
        if not profile:
            raise ValueError("No profile found for that name")
        expected = profile.id
    session = uuid.uuid4().hex
    worker = CameraWorker(config)
    worker.start()
    rows, anchor = [], None
    last = 0
    deadline = time.monotonic() + 90
    print("Only the named participant should be in view. Vary angle and lighting gently. Press Q to cancel.")
    try:
        while len(rows) < count and time.monotonic() < deadline:
            while not worker.events.empty():
                kind, message = worker.events.get_nowait()
                if kind == "error":
                    raise ValueError(message)
            try:
                shot = worker.frames.get(timeout=0.1)
            except __import__("queue").Empty:
                continue
            now = time.monotonic()
            if len(shot.faces) == 1 and shot.faces[0].embedding is not None and now-last >= 0.6:
                vector = shot.faces[0].embedding
                if anchor is None:
                    anchor = vector.copy()
                if float(anchor @ vector) < 0.45:
                    raise ValueError("The face changed during capture. Dataset was not saved.")
                rows.append({"session": session, "split": split, "expected_id": expected, "embedding": vector.tolist()})
                last = now
            cv2.putText(shot.frame, f"{split}: {len(rows)}/{count} - Q cancels", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 230, 170), 2)
            cv2.imshow("Saathi evaluation capture", shot.frame)
            if cv2.waitKey(1) & 255 in (ord("q"), 27):
                raise ValueError("Capture cancelled. Dataset was not saved.")
        if len(rows) < count:
            raise ValueError("Capture timed out. Dataset was not saved.")
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Saved {len(rows)} samples from one {split} session to {path}")
    finally:
        worker.stop()
        worker.thread.join(timeout=3)
        cv2.destroyAllWindows()
