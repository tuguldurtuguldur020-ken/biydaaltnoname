from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import tempfile
import unittest

from platewatch.plates import Detection, PlateTracker, TemporalGate, normalize_plate, select_ocr_candidate
from platewatch.rules import evaluate, rule_plate, restriction_days
from platewatch.settings import DEFAULTS, UB_TIME, load_settings, save_settings
from platewatch.storage import FIELDS, RecordStore


def box(x, y=0):
    return [[x, y], [x + 70, y], [x + 70, y + 20], [x, y + 20]]


class PlateTests(unittest.TestCase):
    def test_confirmation_window_can_cover_a_user_selected_long_sample_gap(self):
        tracker = PlateTracker()
        detection = Detection((10, 10, 90, 35), "1742УБР", .9, .9)
        self.assertFalse(tracker.confirm([detection], 0, window=6)[0].plate)
        self.assertEqual(tracker.confirm([detection], 4, window=6)[0].plate, "1742УБР")

    def test_fast_approaching_car_remains_the_same_track_across_video_samples(self):
        tracker = PlateTracker()
        observations = [(0, (171, 462, 233, 481), "7034УБУ", .666),
                        (.333, (220, 482, 290, 503), "7034ҮБУ", .627),
                        (.667, (297, 515, 385, 540), "7034УБУ", .999)]
        accepted = []
        for clock, bbox, plate, score in observations:
            accepted.extend(d.plate for d in tracker.confirm([Detection(bbox, plate, .9, score)], clock) if d.plate)
        self.assertEqual(accepted, ["7034УБУ"])

    def test_tracking_separates_cars_and_never_counts_a_frame_twice(self):
        tracker = PlateTracker()
        first = Detection((10, 10, 90, 30), "7034УБУ", .9, .95)
        second = Detection((200, 10, 280, 30), "7034УБУ", .9, .95)
        self.assertFalse(tracker.confirm([first, second], 0)[0].plate)
        self.assertFalse(tracker.confirm([first], 0)[0].plate)
        confirmed = tracker.confirm([first], .5)
        self.assertEqual(confirmed[0].plate, "7034УБУ")
        # A newly arriving car with the same text has its own evidence.
        third = Detection((400, 10, 480, 30), "7034УБУ", .9, .95)
        self.assertFalse(tracker.confirm([third], 1)[0].plate)
        self.assertFalse(tracker.confirm([first], 10)[0].plate)

    def test_normalization_preserves_cyrillic_and_leading_zero(self):
        self.assertEqual(normalize_plate("0124 УБӨ"), "0124УБӨ")
        self.assertEqual(normalize_plate("1234 YBA"), "1234УВА")
        self.assertEqual(normalize_plate("1О34 УБА"), "1034УБА")
        self.assertEqual(normalize_plate("7319 ΔΑP"), "7319ДАР")
        for invalid in ["123", "12345УБА", "ДУГААР", "ABCDУБА", "1234???", "1234XYZ"]:
            self.assertEqual(normalize_plate(invalid), "")

    def test_split_ocr_boxes_are_joined_and_noise_ignored(self):
        plate, confidence, _ = select_ocr_candidate([
            (box(100, 2), "УБА", .78), (box(0), "1234", .92)])
        self.assertEqual(plate, "1234УБА")
        self.assertEqual(confidence, .78)
        plate, _, _ = select_ocr_candidate([(box(0, 0), "MNG", .9), (box(0, 40), "1234", .9), (box(80, 40), "УБА", .8)])
        self.assertEqual(plate, "1234УБА")

    def test_temporal_confirmation_cooldown_and_resets(self):
        gate = TemporalGate(confirmations=2, cooldown=20)
        self.assertFalse(gate.ready("1234УБА", 0))
        self.assertTrue(gate.ready("1234УБА", 1))
        gate.mark_saved("1234УБА", 1)
        self.assertFalse(gate.ready("1234УБА", 2))
        self.assertFalse(gate.ready("1234УБА", 21))
        self.assertTrue(gate.ready("1234УБА", 22))
        self.assertTrue(gate.ready("1234УБА", 2, single_image=True))
        gate = TemporalGate(2)
        self.assertFalse(gate.ready("1234УБА", 0))
        self.assertFalse(gate.ready("1234УБА", 15))


class RuleTests(unittest.TestCase):
    def setUp(self):
        self.rules = deepcopy(DEFAULTS["rules"])

    def at(self, day=9, hour=12, minute=0):
        return datetime(2026, 11, day, hour, minute, tzinfo=UB_TIME)

    def test_weekdays_use_last_numeric_digit(self):
        for day, digits in [(9, [1, 6]), (10, [2, 7]), (11, [3, 8]), (12, [4, 9]), (13, [5, 0])]:
            for digit in range(10):
                self.assertEqual(evaluate(f"123{digit}УБА", self.rules, self.at(day)).restricted_now, digit in digits)
        self.assertEqual(evaluate("1231УБА", self.rules, self.at(14)).code, "allowed")

    def test_four_digits_1113_restrict_wednesday_not_friday(self):
        self.assertEqual(rule_plate("1113"), "1113")
        self.assertEqual(restriction_days("1113", self.rules), [2])
        self.assertEqual(restriction_days("1113УБА", self.rules), [2])
        self.assertTrue(evaluate("1113", self.rules, self.at(day=11)).restricted_now)
        self.assertFalse(evaluate("1113", self.rules, self.at(day=13)).restricted_now)

    def test_numeric_schedule_uses_all_ten_final_digits_and_preserves_zero(self):
        for digit in range(10):
            number = "000" + str(digit)
            expected = next(day for day in range(5) if digit in self.rules["weekdays"][str(day)])
            self.assertEqual(restriction_days(number, self.rules), [expected])
        self.assertEqual(rule_plate(" 00 03 "), "0003")
        self.assertEqual(restriction_days("0003", self.rules), [2])
        for invalid in ["111", "11133", "ABC", "1113?"]:
            self.assertEqual(rule_plate(invalid), "")

    def test_time_boundaries_and_utc_conversion(self):
        self.assertFalse(evaluate("1231УБА", self.rules, self.at(hour=7, minute=59)).restricted_now)
        self.assertTrue(evaluate("1231УБА", self.rules, self.at(hour=8)).restricted_now)
        self.assertTrue(evaluate("1231УБА", self.rules, self.at(hour=19, minute=59)).restricted_now)
        result = evaluate("1231УБА", self.rules, self.at(hour=20))
        self.assertEqual(result.code, "outside_hours")
        self.assertTrue(result.restricted_today)
        self.assertTrue(evaluate("1231УБА", self.rules, datetime(2026, 11, 9, 0, tzinfo=timezone.utc)).restricted_now)

    def test_holidays_exempt_zone_and_disabled(self):
        self.rules["holidays"] = ["2026-11-09"]
        self.assertEqual(evaluate("1231УБА", self.rules, self.at()).code, "holiday")
        self.rules["holidays"] = []
        self.rules["exempt_plates"] = ["1231 УБА"]
        self.assertEqual(evaluate("1231УБА", self.rules, self.at()).code, "exempt")
        self.rules["in_zone"] = False
        self.assertEqual(evaluate("1231УБА", self.rules, self.at()).code, "out_of_zone")
        self.rules["mode"] = "off"
        self.assertEqual(evaluate("1231УБА", self.rules, self.at()).code, "off")

    def test_odd_even_and_overnight(self):
        self.rules["mode"] = "odd_even"
        self.assertTrue(evaluate("1234УБА", self.rules, self.at(9)).restricted_now)
        self.assertFalse(evaluate("1234УБА", self.rules, self.at(10)).restricted_now)
        self.rules["even_on_even_date"] = False
        self.assertTrue(evaluate("1234УБА", self.rules, self.at(10)).restricted_now)
        self.rules.update(mode="weekly", start="20:00", end="06:00")
        self.assertTrue(evaluate("1231УБА", self.rules, self.at(hour=23)).restricted_now)
        self.assertFalse(evaluate("1231УБА", self.rules, self.at(hour=12)).restricted_now)


class StorageTests(unittest.TestCase):
    def test_incomplete_csv_rows_do_not_break_search_or_export(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            store = RecordStore(temp)
            first = dict.fromkeys(FIELDS, "")
            first.update(id="first", detected_at="2026-11-13T11:39:00+08:00", plate="6775УБА",
                         source="photo.jpg", source_kind="image", detector_confidence="0.95",
                         ocr_confidence="0.9", rule_code="restricted", rule_label="Хязгаарлалттай")
            second = {**first, "id": "second", "detected_at": "2026-11-13T12:00:00+08:00"}
            with store.csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(first)
                file.write("log\npartial,2026-11-13\n")
                for field in ("id", "detected_at", "plate"):
                    writer.writerow({**first, field: ""})
                writer.writerow(second)
                file.write("log\n")
            original = store.csv_path.read_bytes()

            self.assertEqual(store.records(), [first, second])
            self.assertEqual(store.query("6775", "2026-11-13", True), [second, first])
            self.assertEqual(store.query(day="2026-11-14"), [])
            exported = Path(temp) / "export.csv"
            store.export(exported, store.query())
            with exported.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(list(csv.DictReader(file)), [second, first])
            self.assertEqual(store.csv_path.read_bytes(), original)

    def test_cyrillic_images_metadata_filter_and_export(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            store = RecordStore(Path(temp) / "Монгол бүртгэл")
            at = datetime(2026, 11, 9, 12, tzinfo=UB_TIME)
            rule = evaluate("1231УБА", DEFAULTS["rules"], at)
            frame = np.full((80, 200, 3), 210, dtype=np.uint8)
            row = store.add({"plate": "1231УБА", "source": "=test.jpg", "source_kind": "image", "ocr_confidence": .9, "detector_confidence": .95}, frame, frame[:30], rule, at)
            self.assertEqual(store.query("1231", "2026-11-09", True)[0]["plate"], "1231УБА")
            self.assertEqual(store.query(day="2026-11-10"), [])
            self.assertTrue(store.resolve(row["image_path"]).exists())
            self.assertTrue(store.resolve(row["crop_path"]).exists())
            metadata = json.loads(store.resolve(row["metadata_path"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["time_basis"], "import_processing_time")
            self.assertTrue(metadata["rule"]["restricted_now"])
            export = Path(temp) / "export.csv"
            store.export(export, store.records())
            with export.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(next(csv.DictReader(file))["source"], "'=test.jpg")
            with self.assertRaises(ValueError):
                store.export(store.csv_path, [])

    def test_settings_merge_and_corrupt_file_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            config = load_settings(temp)
            config["rules"]["start"] = "09:00"
            save_settings(config, temp)
            self.assertEqual(load_settings(temp)["rules"]["start"], "09:00")
            self.assertEqual(DEFAULTS["rules"]["start"], "08:00")
            (Path(temp) / "settings.json").write_text("broken", encoding="utf-8")
            self.assertEqual(load_settings(temp)["rules"]["start"], "08:00")


if __name__ == "__main__":
    unittest.main()
