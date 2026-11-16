from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from platewatch.media import open_camera, open_video, read_image


class MediaTests(unittest.TestCase):
    def test_photo_cannot_be_silently_opened_as_a_one_frame_video(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "video.jpg"
            Image.new("RGB", (100, 60)).save(path)
            with self.assertRaisesRegex(RuntimeError, "зургийн файл"):
                open_video(path)

    def test_video_retries_decoder_and_releases_failed_capture(self):
        import cv2
        class Capture:
            released = False
            def __init__(self, valid):
                self.valid = valid
            def isOpened(self):
                return True
            def read(self):
                return (True, np.zeros((60, 100, 3), np.uint8)) if self.valid else (False, None)
            def get(self, prop):
                return 25 if prop == cv2.CAP_PROP_FPS else 50
            def release(self):
                self.released = True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "Монгол.mp4"
            path.touch()
            first, second = Capture(False), Capture(True)
            with patch("cv2.VideoCapture", side_effect=[first, second]):
                capture, frame, info = open_video(path)
            self.assertTrue(first.released)
            self.assertIs(capture, second)
            self.assertEqual(info["duration"], 2)
            capture.release()

    def test_unicode_tiff_and_exif_orientation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "Монгол зураг.tiff"
            Image.new("RGB", (80, 30), (220, 40, 20)).save(path)
            frame = read_image(path)
            self.assertEqual(frame.shape, (30, 80, 3))
            self.assertEqual(tuple(frame[0, 0]), (20, 40, 220))
            path = Path(temp) / "rotated.jpg"
            image = Image.new("RGB", (80, 30), "white")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_image(path).shape, (80, 30, 3))

    def test_camera_retries_backend_when_device_opens_without_frames(self):
        class Capture:
            def __init__(self, valid):
                self.valid = valid
                self.released = False
            def isOpened(self):
                return True
            def read(self):
                return (True, np.zeros((60, 100, 3), np.uint8)) if self.valid else (False, None)
            def release(self):
                self.released = True
        first, second = Capture(False), Capture(True)
        with patch("cv2.VideoCapture", side_effect=[first, second]):
            capture, frame, backend, actual_index = open_camera(0)
        self.assertTrue(first.released)
        self.assertIs(capture, second)
        self.assertEqual(frame.shape, (60, 100, 3))
        self.assertEqual(actual_index, 0)
        second.release()

    def test_cancelled_camera_does_not_open_device(self):
        stop = threading.Event()
        stop.set()
        with patch("cv2.VideoCapture") as capture:
            with self.assertRaises(InterruptedError):
                open_camera(0, cancel=stop)
            capture.assert_not_called()

    def test_unavailable_selected_camera_falls_back_to_real_device(self):
        devices = []
        class Capture:
            def __init__(self, index):
                self.index = index
                self.released = False
            def isOpened(self):
                return self.index == 0
            def read(self):
                return True, np.zeros((60, 100, 3), np.uint8)
            def release(self):
                self.released = True
        def open_device(index, backend):
            device = Capture(index)
            devices.append(device)
            return device
        with patch("cv2.VideoCapture", side_effect=open_device):
            camera, frame, backend, actual = open_camera(2)
        self.assertEqual(actual, 0)
        self.assertEqual(camera.index, 0)
        self.assertTrue(all(device.released for device in devices if device.index == 2))
        camera.release()
