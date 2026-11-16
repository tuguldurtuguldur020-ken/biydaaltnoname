from pathlib import Path
import hashlib
import io
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from platewatch.cyrillic_ocr import CyrillicReader, ensure_cyrillic_model, light_plate, select_reading


class CyrillicOcrTests(unittest.TestCase):
    def test_detected_rows_require_matching_complete_readings(self):
        for conflict, missing in ((False, False), (True, False), (False, True)):
            with self.subTest(conflict=conflict, missing=missing):
                reader = CyrillicReader.__new__(CyrillicReader)
                letters_seen = 0
                def read(image):
                    nonlocal letters_seen
                    if image.shape[:2] == (15, 60):
                        return ("432" if missing else "4321"), .98
                    if image.shape[:2] == (10, 45):
                        letters_seen += 1
                        return ("УБХ" if conflict and letters_seen == 2 else "УБА"), .75
                    return "", 0
                reader.read_line = read
                pair = (np.full((15, 60, 3), 200, np.uint8), np.full((10, 45, 3), 200, np.uint8))
                with patch("platewatch.cyrillic_ocr.two_row_crops", return_value=iter([pair])):
                    result = reader.read_plate(np.full((80, 140, 3), 200, np.uint8), detailed=True)
                self.assertEqual(result[0], "" if conflict or missing else "4321УБА")

    def test_ctc_keeps_repeated_digits_and_weak_letter_confidence(self):
        reader = CyrillicReader.__new__(CyrillicReader)
        reader.characters = ["", "6", "5", "7", "У", "Н", "А"]
        reader.input_name, reader.cancel = "x", None
        # Repeated 6 must survive CTC blank separation, yielding 6657УНА.
        ids = [1, 1, 0, 1, 2, 3, 4, 5, 6, 0]
        probabilities = np.zeros((1, len(ids), len(reader.characters)), np.float32)
        for t, i in enumerate(ids):
            probabilities[0, t, i] = .9 if i != 6 else .2
        reader.session = MagicMock()
        reader.session.run.return_value = [probabilities]
        raw, score = reader.read_line(np.full((20, 100, 3), 127, np.uint8))
        self.assertEqual(raw, "6657УНА")
        self.assertLess(score, .6)
        tensor = reader.session.run.call_args.args[1]["x"]
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.all(tensor[:, :, :, 240:] == 0))

    def test_cancel_is_checked_before_inference(self):
        reader = CyrillicReader.__new__(CyrillicReader)
        reader.cancel = threading.Event()
        reader.cancel.set()
        reader.session = MagicMock()
        with self.assertRaises(InterruptedError):
            reader.read_line(np.zeros((20, 100, 3), np.uint8))
        reader.session.run.assert_not_called()

    def test_physical_plate_excludes_dark_bumper_and_preserves_source(self):
        image = np.full((40, 180, 3), 20, np.uint8)
        image[12:30, 35:145] = 225
        image[15:27, 45:48] = 0
        before = image.copy()
        plate = light_plate(image)
        self.assertLess(plate.shape[0], 30)
        self.assertLess(plate.shape[1], 150)
        self.assertGreater(plate.shape[1], 100)
        self.assertTrue(np.array_equal(image, before))

    def test_same_view_cannot_confirm_itself_and_conflicts_are_rejected(self):
        a = ("6657УНА", .9, "6657YHA")
        b = ("6657УНХ", .95, "6657УНХ")
        self.assertFalse(select_reading([("source", a), ("source", a)], 2)[0])
        self.assertFalse(select_reading([("source", a), ("plate", b)], 1)[0])
        self.assertEqual(select_reading([("source", a), ("plate", a)], 2)[0], "6657УНА")

    def test_square_plate_combines_real_rows_without_inserting_characters(self):
        reader = CyrillicReader.__new__(CyrillicReader)
        reader.read_line = MagicMock(side_effect=[("400", .9), ("4000", .95), ("УЕО", .94),
                                                   ("400", .9), ("УЕО", .9), ("4000", .95), ("УЕО", .94)])
        result = reader.read_plate(np.full((40, 60, 3), 200, np.uint8), fast=True)
        self.assertEqual(result[0], "4000УЕО")

    def test_corrupt_model_download_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("platewatch.cyrillic_ocr.FILES", {"inference.onnx": hashlib.sha256(b"valid").hexdigest()}), \
                 patch("platewatch.cyrillic_ocr.urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    ensure_cyrillic_model(temp)
            self.assertFalse(list(Path(temp).iterdir()))
