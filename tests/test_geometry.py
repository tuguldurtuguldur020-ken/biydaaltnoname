from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
import unittest

import cv2
import numpy as np

from platewatch.geometry import plate_quadrilaterals
from platewatch.recognition import PlateRecognizer
from platewatch.settings import DEFAULTS


def plate_scene():
    plate = np.full((180, 320, 3), 235, np.uint8)
    cv2.putText(plate, "1742", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (15, 15, 15), 5)
    cv2.putText(plate, "YBP", (65, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (15, 15, 15), 4)
    frame = np.full((400, 600, 3), 30, np.uint8)
    src = np.array([[0, 0], [319, 0], [319, 179], [0, 179]], np.float32)
    dst = np.array([[100, 100], [450, 65], [470, 270], [125, 290]], np.float32)
    warped = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(src, dst), (600, 400))
    return np.maximum(frame, warped)


class GeometryTests(unittest.TestCase):
    def test_skewed_two_row_plate_is_rectified_without_changing_source(self):
        frame = plate_scene()
        before = frame.copy()
        found = plate_quadrilaterals(frame)
        self.assertEqual(len(found), 1)
        bbox, crop = found[0]
        self.assertLess(bbox[0], 110)
        self.assertGreater(bbox[2], 450)
        self.assertGreater(crop.shape[1], crop.shape[0])
        self.assertTrue(np.array_equal(frame, before))

    def test_empty_light_panels_are_not_plate_proposals(self):
        frame = np.full((400, 600, 3), 30, np.uint8)
        cv2.rectangle(frame, (100, 100), (450, 300), (230, 230, 230), -1)
        self.assertEqual(plate_quadrilaterals(frame), [])

    def test_large_input_keeps_rectified_crops_bounded(self):
        frame = cv2.resize(plate_scene(), (3000, 2000))
        found = plate_quadrilaterals(frame)
        self.assertTrue(found)
        self.assertLessEqual(max(found[0][1].shape[:2]), 1280)

    def test_detector_miss_is_recovered_only_with_complete_agreeing_ocr(self):
        recognizer = PlateRecognizer.__new__(PlateRecognizer)
        recognizer.settings = deepcopy(DEFAULTS)
        recognizer.class_ids, recognizer.device = [0], "cpu"
        recognizer.model = MagicMock()
        recognizer.model.predict.return_value = [SimpleNamespace(boxes=None)]
        recognizer.read_crop = MagicMock(return_value=("1742УБР", .97, "1742УБР"))
        recognizer._read_crop_text = MagicMock(return_value=("1742УБР", .95, "1742УБР"))
        found = recognizer.detect(plate_scene())
        self.assertEqual(found[0].plate, "1742УБР")
        self.assertEqual(found[0].method, "geometry_ocr")
        self.assertEqual(found[0].detector_confidence, 0)
        recognizer._read_crop_text.return_value = ("1742УБН", .99, "1742УБН")
        self.assertEqual(recognizer.detect(plate_scene()), [])
        recognizer.read_crop.return_value = ("", 0, "174")
        self.assertEqual(recognizer.detect(plate_scene()), [])
