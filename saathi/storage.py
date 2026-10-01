"""Local, parameterized SQLite storage. No images, audio, API keys, or chat logs."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
import uuid

from .recognition import MODEL_ID, normalize


def clean_name(name):
    name = " ".join(unicodedata.normalize("NFKC", name).split())
    if not 1 <= len(name) <= 60 or not any(c.isalpha() for c in name):
        raise ValueError("Enter a name between 1 and 60 characters")
    if any(not (c.isalpha() or unicodedata.category(c).startswith("M") or c in " '-.") for c in name):
        raise ValueError("Names can contain letters, spaces, apostrophes, periods, and hyphens")
    return name


def clean_memory(value):
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", value).strip()
    if len(value) > 1000:
        raise ValueError("Keep approved memory below 1,000 characters")
    return value


@dataclass(frozen=True)
class Profile:
    id: str
    name: str
    memory: str
    created_at: str


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    name_key TEXT NOT NULL UNIQUE, memory TEXT NOT NULL DEFAULT '',
                    consent_at TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings (
                    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                    model_id TEXT NOT NULL, vector TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA secure_delete = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def profiles(self):
        with self.connection() as db:
            return [Profile(r["id"], r["name"], r["memory"], r["created_at"])
                    for r in db.execute("SELECT * FROM profiles ORDER BY name_key")]

    def get(self, person_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM profiles WHERE id=?", (person_id,)).fetchone()
            return Profile(row["id"], row["name"], row["memory"], row["created_at"]) if row else None

    def templates(self):
        with self.connection() as db:
            rows = db.execute("SELECT profile_id, vector FROM embeddings WHERE model_id=?", (MODEL_ID,)).fetchall()
        try:
            return [(r["profile_id"], normalize(json.loads(r["vector"]))) for r in rows]
        except (ValueError, TypeError) as exc:
            raise ValueError("A saved face template is damaged. Review or delete that profile.") from exc

    def enroll(self, name, samples, consent):
        if not consent:
            raise ValueError("Face enrollment requires permission")
        name = clean_name(name)
        samples = [normalize(sample) for sample in samples]
        if not 5 <= len(samples) <= 20:
            raise ValueError("Enrollment requires 5 to 20 face samples")
        person_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection() as db:
                db.execute("INSERT INTO profiles VALUES (?,?,?,?,?,?)",
                           (person_id, name, name.casefold(), "", now, now))
                db.executemany("INSERT INTO embeddings VALUES (?,?,?)",
                               [(person_id, MODEL_ID, json.dumps(v.tolist())) for v in samples])
        except sqlite3.IntegrityError as exc:
            raise ValueError("That name already exists. Use a distinct full name.") from exc
        return person_id

    def update_memory(self, person_id, text):
        with self.connection() as db:
            result = db.execute("UPDATE profiles SET memory=? WHERE id=?", (clean_memory(text), person_id))
            if result.rowcount != 1:
                raise ValueError("This profile no longer exists")

    def delete(self, person_id):
        with self.connection() as db:
            db.execute("DELETE FROM profiles WHERE id=?", (person_id,))
