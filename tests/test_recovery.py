from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from platewatch.rules import evaluate
from platewatch.settings import DEFAULTS, atomic_json, load_settings
from platewatch.storage import FIELDS, RecordStore


class RecoveryTests(unittest.TestCase):
    def test_failed_json_write_preserves_the_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            atomic_json(path, {"valid": True})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                atomic_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_invalid_values_and_nested_rule_types_recover_without_overwriting_file(self):
        incoming = {"frame_stride": "3", "confirmation_frames": -2, "ocr_confidence": "nan",
                    "playback_speed": 0, "save_video": "false", "camera_index": [],
                    "rules": {"start": "bad", "weekdays": [], "holidays": ["bad", "2026-11-17"],
                              "exempt_plates": [None, "1234 УБА", "missing"]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "settings.json"
            atomic_json(path, incoming)
            original = path.read_bytes()
            config = load_settings(temp)
            self.assertEqual(config["frame_stride"], 3)
            self.assertEqual(config["ocr_confidence"], DEFAULTS["ocr_confidence"])
            self.assertEqual(config["confirmation_frames"], DEFAULTS["confirmation_frames"])
            self.assertEqual(config["rules"]["weekdays"], DEFAULTS["rules"]["weekdays"])
            self.assertEqual(config["rules"]["holidays"], ["2026-11-17"])
            self.assertEqual(config["rules"]["exempt_plates"], ["1234УБА"])
            self.assertFalse(config["save_video"])
            self.assertEqual(path.read_bytes(), original)

    def test_concurrent_atomic_json_writes_leave_one_complete_document(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.json"
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: atomic_json(path, {"sequence": i, "payload": "Монгол" * 100}), range(24)))
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(result["sequence"], range(24))
            self.assertEqual(result["payload"], "Монгол" * 100)
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def test_invalid_scores_positions_and_duplicate_ids_do_not_break_history(self):
        with tempfile.TemporaryDirectory() as temp:
            store = RecordStore(temp)
            good = dict.fromkeys(FIELDS, "")
            good.update(id="good", detected_at="2026-11-17T00:00:00+08:00", plate="1742УБР",
                        source="photo.jpg", source_kind="image", detector_confidence="0", ocr_confidence=".98")
            with store.csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(good)
                writer.writerow(good)
                for i, score in enumerate(("bad", "nan", "-1", "2")):
                    writer.writerow({**good, "id": str(i), "ocr_confidence": score})
                writer.writerow({**good, "id": "position", "video_seconds": "oops"})
            original = store.csv_path.read_bytes()
            self.assertEqual(store.records(), [good])
            self.assertEqual(store.csv_path.read_bytes(), original)

    def test_failed_index_append_cleans_evidence_and_keeps_previous_records(self):
        with tempfile.TemporaryDirectory() as temp:
            store = RecordStore(temp)
            image = np.full((60, 180, 3), 200, np.uint8)
            record = {"plate": "1742УБР", "source": "photo.jpg", "source_kind": "image",
                      "ocr_confidence": .98, "detector_confidence": 0}
            rule = evaluate(record["plate"], DEFAULTS["rules"])
            store.add(record, image, image, rule)
            original = store.csv_path.read_bytes()
            files = set(Path(temp).rglob("*.*"))
            with patch("platewatch.storage.csv.DictWriter") as writer:
                writer.return_value.writerow.side_effect = OSError("disk full")
                with self.assertRaises(OSError):
                    store.add(record, image, image, rule)
            self.assertEqual(store.csv_path.read_bytes(), original)
            self.assertEqual(set(Path(temp).rglob("*.*")), files)
            self.assertEqual(len(store.records()), 1)

    def test_evidence_resolver_does_not_escape_its_store(self):
        with tempfile.TemporaryDirectory() as temp:
            store = RecordStore(temp)
            with self.assertRaises(ValueError):
                store.resolve("../other/file.jpg")
            self.assertTrue(store.resolve("evidence/plate.jpg").is_relative_to(Path(temp).resolve()))
