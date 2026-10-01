import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from saathi.config import Config
from saathi.recognition import Enrollment, IdentityGate, Match, Matcher, normalize
from saathi.session import SceneGuard, token_is_current
from saathi.storage import Store, clean_name
from saathi.conversation import CloudAssistant, api_error, parse_intent
from saathi.calibration import calibrate, evaluate, apply_report, template_fingerprint


def vector(axis=0):
    result = np.zeros(128, np.float32)
    result[axis] = 1
    return result


class RecognitionTests(unittest.TestCase):
    def test_bad_embeddings_rejected(self):
        for value in ([1, 2], np.zeros(128), np.full(128, np.nan), np.full(128, np.inf)):
            with self.subTest(value=str(value)[:30]), self.assertRaises(ValueError):
                normalize(value)

    def test_unknown_when_database_empty(self):
        self.assertIsNone(Matcher().match(vector()).person_id)

    def test_same_person_templates_do_not_create_ambiguity(self):
        matcher = Matcher([("a", vector()), ("a", vector()), ("b", vector(1))])
        self.assertEqual(matcher.match(vector()).person_id, "a")

    def test_lookalike_ambiguity_rejected(self):
        matcher = Matcher([("a", vector()), ("b", normalize(vector()+0.01*vector(1)))])
        self.assertEqual(matcher.match(vector()).reason, "ambiguous match")

    def test_unknown_below_threshold(self):
        self.assertIsNone(Matcher([("a", vector())]).match(vector(2)).person_id)

    def test_stability_and_greeting_cooldown(self):
        gate = IdentityGate(3, 60)
        a = [Match("a")]
        self.assertEqual(gate.observe(a, 0), (None, False))
        self.assertEqual(gate.observe(a, 1), (None, False))
        self.assertEqual(gate.observe(a, 2), ("a", True))
        self.assertEqual(gate.observe(a, 3), ("a", False))
        gate.observe([], 4)
        for now in (5, 6, 7):
            result = gate.observe(a, now)
        self.assertEqual(result, ("a", False))
        gate.observe([], 65)
        for now in (66, 67, 68):
            result = gate.observe(a, now)
        self.assertEqual(result, ("a", True))

    def test_multi_face_or_uncertain_match_revokes_identity(self):
        for observations in ([], [Match(None)], [Match("a"), Match("b")]):
            gate = IdentityGate(2)
            gate.observe([Match("a")])
            gate.observe([Match("a")])
            epoch = gate.epoch
            self.assertIsNone(gate.observe(observations)[0])
            self.assertGreater(gate.epoch, epoch)

    def test_new_person_cannot_inherit_streak(self):
        gate = IdentityGate(3)
        gate.observe([Match("a")])
        gate.observe([Match("a")])
        self.assertIsNone(gate.observe([Match("b")])[0])

    def test_enrollment_requires_permission(self):
        with self.assertRaises(ValueError):
            Enrollment("Saiyam", False)

    def test_enrollment_paces_samples_and_finishes(self):
        enrollment = Enrollment("Saiyam", True, count=5, now=0)
        matcher = Matcher()
        self.assertFalse(enrollment.add(vector(), matcher, now=0))
        self.assertFalse(enrollment.add(vector(), matcher, now=0.1))
        self.assertEqual(len(enrollment.samples), 1)
        for now in (1, 2, 3, 4):
            result = enrollment.add(vector(), matcher, now=now)
        self.assertTrue(result)

    def test_enrollment_rejects_person_switch_and_duplicate(self):
        enrollment = Enrollment("Saiyam", True, now=0)
        enrollment.add(vector(), Matcher(), now=0)
        with self.assertRaisesRegex(ValueError, "face changed"):
            enrollment.add(vector(1), Matcher(), now=1)
        with self.assertRaisesRegex(ValueError, "already"):
            Enrollment("Other", True, now=0).add(vector(), Matcher([("a", vector())]), now=0)

    def test_enrollment_timeout(self):
        with self.assertRaisesRegex(ValueError, "timed out"):
            Enrollment("Saiyam", True, timeout=10, now=0).add(vector(), Matcher(), now=11)


class SessionTests(unittest.TestCase):
    def test_unknown_person_swap_invalidates_pending_turn(self):
        guard = SceneGuard()
        guard.observe([vector()])
        epoch = guard.epoch
        self.assertFalse(guard.observe([vector()]))
        self.assertTrue(guard.observe([vector(1)]))
        self.assertGreater(guard.epoch, epoch)

    def test_bad_quality_and_multiple_people_invalidate(self):
        guard = SceneGuard()
        guard.observe([vector()])
        self.assertTrue(guard.observe([None]))
        self.assertFalse(guard.valid)
        guard.observe([vector()])
        self.assertTrue(guard.observe([vector(), vector(1)]))
        self.assertFalse(guard.valid)

    def test_stale_and_out_of_order_responses_discarded(self):
        self.assertTrue(token_is_current((1, 2, 3), (1, 2, 3), 0.1, True))
        self.assertFalse(token_is_current((1, 2, 3), (1, 2, 4), 0.1, True))
        self.assertFalse(token_is_current((1, 2, 3), (1, 2, 3), 2.1, True))
        self.assertFalse(token_is_current((1, 2, 3), (1, 2, 3), 0.1, False))


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "db.sqlite3")

    def test_enrollment_persists_and_deletion_cascades(self):
        person = self.store.enroll("Saiyam", [vector()]*5, True)
        self.store.update_memory(person, "I study Python")
        reopened = Store(self.store.path)
        self.assertEqual(reopened.get(person).memory, "I study Python")
        self.assertEqual(len(reopened.templates()), 5)
        reopened.delete(person)
        self.assertEqual(reopened.templates(), [])
        self.assertEqual(reopened.profiles(), [])

    def test_duplicate_name_does_not_partial_write(self):
        self.store.enroll("Saiyam", [vector()]*5, True)
        with self.assertRaises(ValueError):
            self.store.enroll(" SAIYAM ", [vector(1)]*5, True)
        self.assertEqual(len(self.store.profiles()), 1)
        self.assertEqual(len(self.store.templates()), 5)

    def test_consent_and_sample_count(self):
        for consent, samples in ((False, [vector()]*5), (True, [vector()])):
            with self.assertRaises(ValueError):
                self.store.enroll("Saiyam", samples, consent)
        self.assertEqual(self.store.profiles(), [])

    def test_memory_isolated_and_missing_user_cannot_write(self):
        a = self.store.enroll("Saiyam", [vector()]*5, True)
        b = self.store.enroll("Vikas", [vector(1)]*5, True)
        self.store.update_memory(a, "Python")
        self.assertEqual(self.store.get(b).memory, "")
        with self.assertRaises(ValueError):
            self.store.update_memory("missing", "note")
        with self.assertRaises(ValueError):
            self.store.update_memory(a, "x"*1001)

    def test_names_accept_unicode_and_reject_injection(self):
        self.assertEqual(clean_name("  सैयम  कुमार "), "सैयम कुमार")
        self.assertEqual(clean_name("O'Connor"), "O'Connor")
        for name in ("", "123", "a; DROP TABLE profiles", "x"*61):
            with self.assertRaises(ValueError):
                clean_name(name)


class ConversationTests(unittest.TestCase):
    def test_introduction_and_memory_are_explicit(self):
        self.assertEqual(parse_intent("My name is Saiyam.").value, "Saiyam")
        self.assertEqual(parse_intent("remember that I study Python").kind, "remember")
        self.assertEqual(parse_intent("I study Python").kind, "chat")
        self.assertEqual(parse_intent("forget my memory").kind, "forget_memory")

    def test_chat_payload_has_no_face_or_persistent_response(self):
        from saathi.storage import Profile
        client = Mock()
        client.responses.create.return_value.output_text = "Hi Saiyam."
        cloud = CloudAssistant("", Config(), client)
        history = [{"role": "user", "content": "hello"}] * 20
        self.assertEqual(cloud.reply(Profile("a", "Saiyam", "Python", ""), history, "Hi"), "Hi Saiyam.")
        kwargs = client.responses.create.call_args.kwargs
        self.assertFalse(kwargs["store"])
        self.assertEqual(len(kwargs["input"]), 11)
        self.assertIn("Python", kwargs["instructions"])
        self.assertNotIn("embedding", json.dumps(kwargs))
        self.assertNotIn("image", json.dumps(kwargs["input"]))

    def test_cloud_errors_do_not_echo_secrets(self):
        client = Mock()
        client.responses.create.side_effect = RuntimeError("sk-private-do-not-print")
        with self.assertRaises(RuntimeError) as error:
            CloudAssistant("", Config(), client).reply(None, [], "hi")
        self.assertNotIn("sk-private", str(error.exception))
        error_type = type("AuthenticationError", (Exception,), {})
        self.assertIn("key was rejected", api_error(error_type()))

    def test_missing_key_and_blank_input(self):
        with self.assertRaises(ValueError):
            CloudAssistant("", Config())
        for value in ("", " ", "x"*2001):
            with self.assertRaises(ValueError):
                parse_intent(value)


class ConfigTests(unittest.TestCase):
    def test_unsafe_or_invalid_values(self):
        for kwargs in ({"target_fps": 0}, {"stable_frames": 1.5}, {"camera_index": True},
                       {"recognition_threshold": float("nan")}, {"min_brightness": 250, "max_brightness": 200},
                       {"microphone_index": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)


class CalibrationTests(unittest.TestCase):
    def dataset(self):
        rng = np.random.default_rng(12)
        rows = []
        for split in ("tune", "test"):
            for i in range(20):
                expected = "a" if i % 2 == 0 else None
                v = normalize(vector(0 if expected else 1) + rng.normal(0, 0.015, 128))
                rows.append({"split": split, "session": split, "expected_id": expected, "embedding": v})
        return rows

    def test_heldout_metrics_and_apply_preserve_other_config(self):
        templates = [("a", vector())]
        report = calibrate(self.dataset(), templates)
        self.assertEqual(report["held_out_test"]["correct"], 20)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"camera_index": 2}', encoding="utf-8")
            apply_report(report, templates, path)
            self.assertEqual(Config.load(path).camera_index, 2)

    def test_leakage_unknown_coverage_and_missing_profiles_rejected(self):
        rows = self.dataset()
        cases = [rows[:20], [{**r, "session": "same"} for r in rows],
                 [{**r, "expected_id": "missing"} for r in rows],
                 rows + [rows[0]], [{**r, "expected_id": "a"} for r in rows]]
        for values in cases:
            with self.assertRaises(ValueError):
                calibrate(values, [("a", vector())])

    def test_changed_profiles_and_failed_heldout_block_apply(self):
        report = calibrate(self.dataset(), [("a", vector())])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            with self.assertRaises(ValueError):
                apply_report(report, [("b", vector())], path)
            report["held_out_test"]["wrong_identity"] = 1
            with self.assertRaises(ValueError):
                apply_report(report, [("a", vector())], path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
