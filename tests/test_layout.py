from unittest import TestCase
import cv2
import numpy as np
from platewatch.layout import two_row_crops


class PlateLayoutTests(TestCase):
    def fixture(self, numbers="1234"):
        image = np.full((160, 220, 3), 60, np.uint8)
        image[30:140, 55:215] = 210
        cv2.putText(image, numbers, (67, 78), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
        cv2.putText(image, "ABC", (87, 121), cv2.FONT_HERSHEY_SIMPLEX, .8, (10, 10, 10), 2)
        return image

    def test_slanted_rows_are_separated_from_an_oversized_detector_crop(self):
        source = self.fixture()
        for angle in (0, 12, -10):
            with self.subTest(angle=angle):
                image = cv2.warpAffine(source, cv2.getRotationMatrix2D((110, 80), angle, 1),
                                       (220, 160), borderValue=(60, 60, 60))
                before = image.copy()
                pairs = list(two_row_crops(image))
                self.assertTrue(pairs)
                self.assertLessEqual(len(pairs), 2)
                for number, letters in pairs:
                    self.assertGreater(number.shape[1], 2 * number.shape[0])
                    self.assertGreater(letters.shape[1], 1.5 * letters.shape[0])
                    self.assertLess(number.shape[1], 720)
                np.testing.assert_array_equal(image, before)

    def test_missing_character_is_not_replaced_by_background_trim(self):
        self.assertEqual(list(two_row_crops(self.fixture("123"))), [])

    def test_empty_panel_and_single_text_row_do_not_become_two_rows(self):
        image = np.full((120, 180, 3), 220, np.uint8)
        self.assertEqual(list(two_row_crops(image)), [])
        cv2.putText(image, "1234", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
        self.assertEqual(list(two_row_crops(image)), [])

    def test_tiny_and_single_line_crops_are_skipped(self):
        for shape in ((0, 40, 3), (15, 35, 3), (40, 240, 3)):
            self.assertEqual(list(two_row_crops(np.zeros(shape, np.uint8))), [])
