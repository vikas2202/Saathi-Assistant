"""Pure recognition and enrollment logic, independent of cameras and the UI."""
from dataclasses import dataclass
import time
import numpy as np

EMBEDDING_SIZE = 128
MODEL_ID = "opencv-sface-2021dec"


def normalize(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != EMBEDDING_SIZE or not np.isfinite(vector).all():
        raise ValueError("Invalid face embedding")
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("Empty face embedding")
    return vector / length


@dataclass(frozen=True)
class Match:
    person_id: str | None
    score: float = 0.0
    margin: float = 0.0
    reason: str = "unknown"


class Matcher:
    def __init__(self, templates=(), threshold=0.5, margin=0.08):
        self.threshold = threshold
        self.margin = margin
        self.ids = [person_id for person_id, _ in templates]
        self.matrix = (np.stack([normalize(v) for _, v in templates])
                       if self.ids else np.empty((0, EMBEDDING_SIZE), np.float32))

    def scores(self, embedding):
        query = normalize(embedding)
        if not self.ids:
            return []
        values = np.clip(self.matrix @ query, -1, 1)
        # Rank identities, not individual samples: two samples of one person
        # must never be treated as competing people in the ambiguity check.
        grouped = {}
        for person_id, score in zip(self.ids, values):
            grouped[person_id] = max(grouped.get(person_id, -1.0), float(score))
        return sorted(grouped.items(), key=lambda row: row[1], reverse=True)

    def match(self, embedding):
        ranked = self.scores(embedding)
        if not ranked:
            return Match(None)
        person_id, best = ranked[0]
        gap = best - ranked[1][1] if len(ranked) > 1 else best + 1
        if best < self.threshold:
            return Match(None, best, gap, "below threshold")
        if gap < self.margin:
            return Match(None, best, gap, "ambiguous match")
        return Match(person_id, best, gap, "matched")


class IdentityGate:
    """Require consecutive matches; invalidate immediately on uncertainty."""
    def __init__(self, stable_frames=5, cooldown=90):
        self.required = stable_frames
        self.cooldown = cooldown
        self.candidate = None
        self.count = 0
        self.active = None
        self.epoch = 0
        self.last_greeting = {}

    def observe(self, matches, now=None):
        now = time.monotonic() if now is None else now
        candidate = matches[0].person_id if len(matches) == 1 else None
        old = self.active
        if candidate is None:
            self.candidate, self.count, self.active = None, 0, None
        else:
            self.count = self.count + 1 if candidate == self.candidate else 1
            self.candidate = candidate
            self.active = candidate if self.count >= self.required else None
        if old != self.active:
            self.epoch += 1
        greet = False
        if self.active and old != self.active:
            last = self.last_greeting.get(self.active, float("-inf"))
            if now - last >= self.cooldown:
                self.last_greeting[self.active] = now
                greet = True
        return self.active, greet

    def reset(self):
        self.candidate, self.active, self.count = None, None, 0
        self.epoch += 1


class Enrollment:
    def __init__(self, name, consent, count=10, timeout=35, now=None):
        if not consent:
            raise ValueError("Permission is required before saving a face profile")
        self.name = name
        self.required = count
        self.deadline = (time.monotonic() if now is None else now) + timeout
        self.samples = []
        self.last_sample = float("-inf")

    def add(self, embedding, matcher, now=None):
        now = time.monotonic() if now is None else now
        if now > self.deadline:
            raise ValueError("Enrollment timed out. Improve the lighting and try again.")
        if now - self.last_sample < 0.4:
            return False
        sample = normalize(embedding)
        ranked = matcher.scores(sample)
        if ranked and ranked[0][1] >= matcher.threshold:
            raise ValueError("This face may already have a profile. Use that profile or review it first.")
        if self.samples and float(self.samples[0] @ sample) < 0.45:
            raise ValueError("The face changed during enrollment. Please start again with one person.")
        self.samples.append(sample)
        self.last_sample = now
        return len(self.samples) >= self.required
