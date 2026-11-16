from unittest.mock import patch, MagicMock
import unittest
import numpy as np
import cv2
from copy import deepcopy
from types import SimpleNamespace
from platewatch.settings import DEFAULTS
from platewatch.ocr import text_regions
from platewatch.plates import select_ocr_candidate


class OcrCropTests(unittest.TestCase):
    def test_empty_legacy_fallback_preserves_a_confirmed_detailed_reading(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.cyrillic_reader = MagicMock()
        recognizer.cyrillic_reader.read_plate.return_value = ("4321УБА", .6, "4321УБА")
        recognizer._read_crop_text = MagicMock(return_value=("", 0, ""))
        self.assertEqual(recognizer.read_crop(np.full((80, 140, 3), 200, np.uint8), detailed=True)[0], "4321УБА")

    def test_model_agreement_survives_a_conflicting_higher_score_from_one_model(self):
        from platewatch.recognition import merge_detections
        from platewatch.plates import Detection
        agreed = Detection((350, 383, 630, 435), "7319ДАР", .95, .51, reading_support=2)
        conflicting = Detection((393, 384, 584, 437), "7319ААР", .97, .70)
        self.assertEqual(merge_detections([agreed, conflicting])[0].plate, "7319ДАР")

    def test_small_square_video_plate_is_read_instead_of_skipped(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids, recognizer.device = [0], "cpu"
        box = SimpleNamespace(xyxy=np.array([[50, 40, 81, 63]]), conf=[.9])
        recognizer.model = MagicMock()
        recognizer.model.predict.return_value = [SimpleNamespace(boxes=[box])]
        recognizer.read_crop = MagicMock(return_value=("4000УЕО", .9, "4000УЕО"))
        result = recognizer.detect_video_frame(np.full((100, 200, 3), 200, np.uint8))
        self.assertEqual(result[0].plate, "4000УЕО")
        recognizer.read_crop.assert_called_once()

    def test_disagreement_of_independent_recognizers_does_not_choose_wrong_last_letter(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.cyrillic_reader = MagicMock()
        recognizer.cyrillic_reader.read_plate.return_value = ("6657УНА", .6, "6657YHA")
        recognizer._read_crop_text = MagicMock(return_value=("6657УНХ", .95, "6657УНХ"))
        self.assertFalse(recognizer.read_crop(np.full((20, 100, 3), 200, np.uint8), detailed=True)[0])

    def test_detailed_reading_rejects_conflicting_preprocessing_results(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.reader = MagicMock()
        box = [[0, 0], [220, 0], [220, 70], [0, 70]]
        variants = [np.full((80, 250), 255, np.uint8)] * 2
        recognizer.reader.recognize.side_effect = [[(box, "7034УБУ", .99)], [(box, "1703УБУ", .99)]]
        with patch("platewatch.recognition.crop_variants", return_value=iter(variants)):
            self.assertFalse(recognizer._read_crop_text(np.zeros((80, 250, 3), np.uint8), detailed=True)[0])

    def test_detailed_reading_requires_agreement_of_distinct_variants(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.reader = MagicMock()
        box = [[0, 0], [220, 0], [220, 70], [0, 70]]
        recognizer.reader.recognize.side_effect = [[(box, "7034УБУ", .99)], [(box, "7034УБУ", .91)]]
        plate, confidence, _ = recognizer.read_crop(np.full((80, 250, 3), 255, np.uint8), detailed=True)
        self.assertEqual(plate, "7034УБУ")
        self.assertEqual(confidence, .91)

    def test_detector_retries_at_smaller_input_after_allocation_failure(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids, recognizer.device = [0], "cpu"
        recognizer.model = MagicMock()
        recognizer.model.predict.side_effect = [RuntimeError("not enough memory"), [SimpleNamespace(boxes=None)]]
        self.assertEqual(recognizer.detect(np.zeros((100, 220, 3), np.uint8)), [])
        self.assertEqual([call.kwargs["imgsz"] for call in recognizer.model.predict.call_args_list], [640, 320])

    def test_detector_does_not_hide_an_unrelated_runtime_error(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids, recognizer.device = [0], "cpu"
        recognizer.model = MagicMock()
        recognizer.model.predict.side_effect = RuntimeError("shape mismatch")
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            recognizer.detect(np.zeros((100, 220, 3), np.uint8))
        self.assertEqual(recognizer.model.predict.call_count, 1)

    def test_light_plate_band_removes_dark_padding_without_modifying_source(self):
        from platewatch.recognition import plate_text_band
        crop = np.full((32, 140, 3), 20, np.uint8)
        crop[10:27, 4:136] = 220
        before = crop.copy()
        band = plate_text_band(crop)
        self.assertLess(band.shape[0], crop.shape[0])
        self.assertGreaterEqual(band.shape[0], 17)
        self.assertTrue(np.array_equal(crop, before))

    def test_detailed_preprocessing_preserves_source_and_bounds_all_variants(self):
        from platewatch.recognition import crop_variants
        crop = np.full((8, 52, 3), (100, 130, 160), np.uint8)
        before = crop.copy()
        variants = list(crop_variants(crop, detailed=True))
        self.assertEqual(len(variants), 4)
        self.assertTrue(np.array_equal(crop, before))
        self.assertGreater(variants[0].shape[0], crop.shape[0])
        self.assertTrue(all(v.ndim == 2 and v.dtype == np.uint8 and v.shape[1] <= 1800 for v in variants))

    def test_emblem_does_not_force_an_extra_character_between_digits_and_letters(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.reader = MagicMock()
        box = [[0, 0], [80, 0], [80, 30], [0, 30]]
        recognizer.reader.recognize.side_effect = [[(box, "1234", .93)], [(box, "УБА", .87)]]
        plate, confidence, raw = recognizer.read_split_crop(np.full((50, 240), 180, np.uint8))
        self.assertEqual(plate, "1234УБА")
        self.assertAlmostEqual(confidence, .87)
        self.assertEqual(recognizer.reader.recognize.call_args_list[0].kwargs["allowlist"], "0123456789")

    def test_split_reading_does_not_invent_a_missing_digit(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.reader = MagicMock()
        box = [[0, 0], [80, 0], [80, 30], [0, 30]]
        recognizer.reader.recognize.side_effect = [[(box, "123", .93)], [(box, "УБА", .87)]] * 3
        self.assertEqual(recognizer.read_split_crop(np.full((50, 240), 180, np.uint8))[0], "")

    def test_tile_boxes_are_mapped_to_the_original_frame(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids = [0]
        recognizer.device = "cpu"
        recognizer.model = MagicMock()
        box = SimpleNamespace(xyxy=np.array([[10, 20, 70, 40]]), conf=[.8])
        recognizer.model.predict.side_effect = [[SimpleNamespace(boxes=None)] for _ in range(4)] + [[SimpleNamespace(boxes=[box])]]
        recognizer.read_crop = lambda crop, **kwargs: ("1234УБА", .95, "1234УБА")
        found = recognizer.detect_detailed(np.full((640, 360, 3), 200, np.uint8))
        self.assertEqual(len(found), 1)
        self.assertGreaterEqual(found[0].bbox[0], 144)
        self.assertGreaterEqual(found[0].bbox[1], 256)
        self.assertEqual(recognizer.model.predict.call_count, 5)

    def test_tile_overlap_keeps_best_reading_and_separate_cars(self):
        from platewatch.recognition import merge_detections
        from platewatch.plates import Detection
        incomplete = Detection((10, 10, 45, 20), "", .9, 0)
        readable = Detection((8, 8, 70, 24), "1234УБА", .8, .95)
        another = Detection((100, 10, 160, 25), "5678УБВ", .7, .9)
        result = merge_detections([incomplete, readable, another])
        self.assertEqual({d.plate for d in result}, {"1234УБА", "5678УБВ"})

    def test_partial_digit_box_is_widened_to_include_letters(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids = [0]
        recognizer.device = "cpu"
        box = SimpleNamespace(xyxy=np.array([[0, 235, 452, 391]]), conf=[.4])
        recognizer.model = MagicMock()
        recognizer.model.predict.return_value = [SimpleNamespace(boxes=[box])]
        recognizer.read_crop = lambda crop: ("9638УЕН", .98, "9638УЕН") if crop.shape[1] == 720 else ("", 0, "9638")
        result = recognizer.detect(np.full((720, 720, 3), 200, np.uint8))
        self.assertEqual(result[0].plate, "9638УЕН")
        self.assertEqual(result[0].bbox[2], 720)
    def test_prominent_text_region_excludes_small_country_label(self):
        image = np.full((120, 480), 255, np.uint8)
        cv2.putText(image, "MNG", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .4, 0, 1)
        cv2.putText(image, "1234 ABA", (65, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 0, 3)
        regions = text_regions(image)
        self.assertTrue(regions)
        self.assertTrue(all(region[2] > 20 for region in regions))

    def test_no_detected_characters_uses_whole_plate_in_recognizer(self):
        from platewatch.recognition import PlateRecognizer
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.reader = MagicMock()
        recognizer.reader.recognize.return_value = [([[0, 0], [220, 0], [220, 70], [0, 70]], "1234УБА", .95)]
        plate, confidence, raw = recognizer.read_crop(np.full((80, 250, 3), 255, np.uint8))
        self.assertEqual(plate, "1234УБА")
        self.assertGreater(confidence, .9)
        recognizer.reader.readtext.assert_not_called()
        regions = recognizer.reader.recognize.call_args.kwargs["horizontal_list"]
        self.assertIsInstance(regions, list)
        self.assertTrue(regions)
        self.assertEqual(regions[0][0], 0)

    def test_split_two_line_plate_can_be_combined(self):
        items = [([[0, 0], [180, 0], [180, 40], [0, 40]], "1234", .9),
                 ([[25, 55], [155, 55], [155, 95], [25, 95]], "УБА", .85)]
        self.assertEqual(select_ocr_candidate(items)[0], "1234УБА")

    def test_country_logo_gap_keeps_digits_and_letters_separate(self):
        image = np.full((100, 400), 255, np.uint8)
        cv2.putText(image, "1234", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 0, 3)
        cv2.circle(image, (215, 52), 10, 0, 1)
        cv2.putText(image, "ABA", (260, 76), cv2.FONT_HERSHEY_SIMPLEX, 1.7, 0, 3)
        regions = text_regions(image)
        self.assertGreaterEqual(len(regions), 2)
        self.assertTrue(all(not (box[0] < 215 < box[1]) for box in regions))

    def test_large_oval_country_logo_is_not_a_character(self):
        image = np.full((156, 720), 255, np.uint8)
        cv2.putText(image, "9638", (55, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        cv2.ellipse(image, (380, 95), (36, 24), 0, 0, 360, 0, 2)
        cv2.putText(image, "YEH", (440, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        regions = text_regions(image)
        self.assertGreaterEqual(len(regions), 2)
        self.assertTrue(all(not (box[0] < 380 < box[1]) for box in regions))
