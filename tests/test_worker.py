from copy import deepcopy
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image

from platewatch.plates import Detection
from platewatch.settings import DEFAULTS
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


class FakeRecognizer:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return [Detection((10, 10, 180, 60), "1234УБА", .95, .92, "1234 УБА")]


class WorkerIntegrationTests(unittest.TestCase):
    def test_camera_honors_the_selected_detailed_recognition_mode(self):
        class ModeRecognizer(FakeRecognizer):
            def __init__(self):
                self.detailed_called = False
            def detect_detailed(self, frame):
                self.detailed_called = True
                return []
        with tempfile.TemporaryDirectory() as temp:
            settings = deepcopy(DEFAULTS)
            settings["recognition_mode"] = "detailed"
            worker = DetectionWorker(settings, RecordStore(temp))
            worker.recognizer = ModeRecognizer()
            worker.process(np.zeros((100, 220, 3), np.uint8), "camera", "Camera 0", 0)
            self.assertTrue(worker.recognizer.detailed_called)

    def test_recorded_unsampled_frames_do_not_reuse_old_plate_boxes(self):
        class Recorder:
            def __init__(self):
                self.frames = []
            def isOpened(self):
                return True
            def write(self, frame):
                self.frames.append(frame.copy())
            def release(self):
                pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "record.avi"
            self.make_video(path, count=4, fps=10)
            settings = deepcopy(DEFAULTS)
            settings.update(frame_stride=2, _video_review=True, save_video=True, adaptive_sampling=False)
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            recorder = Recorder()
            with patch("platewatch.worker.PlateRecognizer", FakeRecognizer), patch("cv2.VideoWriter", return_value=recorder):
                worker.start("video", path)
                worker.thread.join(5)
            self.assertEqual(len(recorder.frames), 4)
            skipped = recorder.frames[1]
            self.assertTrue(np.all(skipped[:, :, 0] == skipped[:, :, 1]))
            self.assertTrue(np.all(skipped[:, :, 1] == skipped[:, :, 2]))
            self.assertGreater(float(recorder.frames[0].std()), 10)

    def test_pending_plate_gets_nearby_samples_before_readable_view_is_missed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "brief.avi"
            self.make_video(path, count=12, fps=10)
            for adaptive, expected_saved in [(True, 1), (False, 0)]:
                settings = deepcopy(DEFAULTS)
                settings.update(frame_stride=10, _video_review=True, adaptive_sampling=adaptive)
                worker = DetectionWorker(settings, RecordStore(root / str(adaptive)))
                class BriefRecognizer(FakeRecognizer):
                    def detect(self, frame):
                        # Only the first two source frames contain the plate.
                        return super().detect(frame) if float(frame.mean()) < 25 else []
                with patch("platewatch.worker.PlateRecognizer", BriefRecognizer):
                    worker.start("video", path)
                    worker.thread.join(5)
                self.assertFalse(worker.alive)
                self.assertEqual(worker.saved_count, expected_saved)
                if adaptive:
                    self.assertEqual(worker.store.records()[0]["video_seconds"], "0.1")

    def test_manual_partial_results_compare_every_car_before_saving(self):
        class MultiCarRecognizer(FakeRecognizer):
            def detect_detailed(self, frame):
                return [Detection((10, 10, 90, 35), "6657УНА", .9, .9), Detection((130, 10, 210, 35))]
            def detect_video_frame(self, frame):
                return [Detection((10, 10, 90, 35), "6657УНА", .9, .9),
                        Detection((130, 10, 210, 35), "7034УБУ", .9, .9)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "partial.avi"
            self.make_video(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", MultiCarRecognizer):
                worker.check_frame(np.full((100, 220, 3), 200, np.uint8), "video", str(path), .2)
                worker.frame_worker.thread.join(5)
            self.assertFalse(worker.alive)
            self.assertEqual({r["plate"] for r in worker.store.records()}, {"6657УНА", "7034УБУ"})
            result = [v for k, v in worker.events.queue if k == "frame_result"][0]
            self.assertEqual(len([d for d in result["detections"] if d.plate]), 2)
            self.assertTrue(result["from_neighbor"])

    def test_manual_first_frame_misread_cannot_be_saved_without_neighbor_support(self):
        class CorrectingRecognizer(FakeRecognizer):
            def detect_detailed(self, frame):
                return [Detection((10, 10, 180, 60), "1703УБУ", .9, .99)]
            def detect_video_frame(self, frame):
                return [Detection((10, 10, 180, 60), "7034УБУ", .9, .99)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "misread.avi"
            self.make_video(path)
            settings = deepcopy(DEFAULTS)
            settings["neighbor_frames"] = 3
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", CorrectingRecognizer):
                worker.check_frame(np.full((100, 220, 3), 200, np.uint8), "video", str(path), .2)
                worker.frame_worker.thread.join(5)
            self.assertEqual([r["plate"] for r in worker.store.records()], ["7034УБУ"])

    def test_blurred_manual_frame_uses_two_matching_real_neighbor_frames(self):
        class NeighborRecognizer(FakeRecognizer):
            def detect_detailed(self, frame):
                return [Detection((10, 10, 140, 60), "", .9, 0, "unreadable")]
            def detect_video_frame(self, frame):
                return super().detect(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "neighbors.avi"
            self.make_video(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            original = np.full((100, 220, 3), 245, np.uint8)
            with patch("platewatch.worker.PlateRecognizer", NeighborRecognizer):
                self.assertTrue(worker.check_frame(original, "video", str(path), .2))
                worker.frame_worker.thread.join(8)
            self.assertFalse(worker.alive)
            results = [v for k, v in worker.events.queue if k == "frame_result"]
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["from_neighbor"])
            self.assertTrue(np.array_equal(results[0]["requested_original"], original))
            self.assertFalse(np.array_equal(results[0]["original"], original))
            self.assertAlmostEqual(results[0]["seconds"], .30)
            self.assertEqual(worker.frame_worker.saved_count, 1)

    def test_stop_cancels_a_manual_check_and_releases_the_shared_model(self):
        import threading
        entered = threading.Event()
        class WaitingRecognizer(FakeRecognizer):
            def __init__(self, settings, report, cancel):
                self.cancel = cancel
            def detect_detailed(self, frame):
                entered.set()
                self.cancel.wait(3)
                raise InterruptedError()
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            with patch("platewatch.worker.PlateRecognizer", WaitingRecognizer):
                worker.check_frame(np.zeros((100, 220, 3), np.uint8), "video", "source.mp4", 1)
                self.assertTrue(entered.wait(3))
                worker.stop()
                worker.frame_worker.thread.join(5)
            self.assertFalse(worker.alive)
            self.assertNotIn("frame_result", [k for k, _ in worker.events.queue])

    def test_pausing_video_does_not_pause_background_analysis(self):
        class SlowRecognizer(FakeRecognizer):
            def detect(self, frame):
                time.sleep(.025)
                return super().detect(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "pause.avi"
            self.make_video(path)
            settings = deepcopy(DEFAULTS)
            settings["frame_stride"] = 2
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", SlowRecognizer):
                worker.start("video", path)
                try:
                    worker.frames.get(timeout=3)
                    worker.pause_event.set()
                    deadline = time.monotonic() + 4
                    while worker.processed < 20 and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertEqual(worker.processed, 20)
                    self.assertTrue(worker.pause_event.is_set())
                    self.assertTrue(worker.alive)
                finally:
                    worker.stop()
                    worker.thread.join(5)

    def test_finished_source_can_recheck_exact_frame_with_shared_cancelable_model(self):
        calls = []
        class CancelableRecognizer(FakeRecognizer):
            def __init__(self, settings, report, cancel):
                self.cancel = cancel
            def detect_detailed(self, frame):
                if self.cancel.is_set():
                    raise InterruptedError()
                calls.append(frame.copy())
                return super().detect(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "input.png"
            Image.new("RGB", (220, 100), "white").save(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", CancelableRecognizer):
                worker.start("image", [path])
                worker.thread.join(5)
                model = worker.recognizer
                self.assertTrue(worker.stop_event.is_set())
                rgb = np.full((100, 220, 3), (11, 22, 33), np.uint8)
                self.assertTrue(worker.check_frame(rgb, "video", "downloaded.mp4", 11.0))
                worker.frame_worker.thread.join(5)
            self.assertFalse(worker.alive)
            self.assertIs(worker.frame_worker.recognizer, model)
            self.assertEqual(len(calls), 1)
            self.assertEqual(tuple(calls[0][0, 0]), (33, 22, 11))
            self.assertEqual(worker.store.records()[-1]["video_seconds"], "11.0")
            self.assertIn("frame_result", [k for k, _ in worker.events.queue])

    def test_detailed_video_checks_every_frame_and_retains_one_matching_preview(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "review.avi"
            self.make_video(path, count=12, fps=10)
            settings = deepcopy(DEFAULTS)
            settings.update(frame_stride=1, _video_review=True)
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", FakeRecognizer):
                worker.start("video", path)
                worker.thread.join(5)
            self.assertFalse(worker.alive)
            self.assertEqual(worker.processed, 12)
            self.assertIsNone(worker.capture_thread)
            self.assertEqual(worker.review_frames.qsize(), 1)
            annotated, raw, seconds = worker.review_frames.get_nowait()
            self.assertEqual(annotated.shape, raw.shape)
            self.assertAlmostEqual(seconds, 1.1)
            self.assertEqual(worker.saved_count, 1)

    def test_review_honors_stride_and_runtime_changes_without_restarting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "stride.avi"
            self.make_video(path, count=12, fps=10)
            settings = deepcopy(DEFAULTS)
            settings.update(frame_stride=5, _video_review=True, adaptive_sampling=False)
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            samples = []
            class ChangingRecognizer(FakeRecognizer):
                def detect(self, frame):
                    samples.append(float(frame.mean()))
                    if len(samples) == 2:
                        worker.settings["frame_stride"] = 2
                    return super().detect(frame)
            with patch("platewatch.worker.PlateRecognizer", ChangingRecognizer):
                worker.start("video", path)
                worker.thread.join(5)
            self.assertFalse(worker.alive)
            # Samples 0, 5, then 7, 9, 11: changing speed must not restart.
            self.assertEqual(len(samples), 5)
            self.assertAlmostEqual(worker.review_frames.get_nowait()[2], 1.1)
            self.assertEqual(worker.saved_count, 1)

    def test_conflicting_video_readings_never_reach_display_or_saved_records(self):
        from platewatch.plates import TemporalGate
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            worker.gate = TemporalGate(1, 20)
            worker.recognizer = FakeRecognizer()
            readings = [("7034УБУ", .95), ("7034УБУ", .98), ("1034УБУ", .57),
                        ("1703УБУ", .79), ("1032УБУ", .98), ("7034УБУ", .99)]
            frame = np.full((100, 220, 3), 200, np.uint8)
            shown = []
            for i, (plate, score) in enumerate(readings):
                worker.recognizer.detect = lambda f, p=plate, s=score: [Detection((10, 10, 180, 60), p, .9, s, p)]
                detections, _ = worker.process(frame, "video", "source.mp4", i / 5)
                shown.extend(d.plate for d in detections if d.plate)
            self.assertEqual(set(shown), {"7034УБУ"})
            self.assertEqual([r["plate"] for r in worker.store.records()], ["7034УБУ"])
            displayed = [value[0].plate for kind, value in worker.events.queue if kind == "detection"]
            self.assertEqual(set(displayed), {"7034УБУ"})

    def test_manual_and_background_checks_never_use_the_model_concurrently(self):
        active = []
        peak = []
        class SerializedRecognizer(FakeRecognizer):
            def detect(self, frame):
                active.append(True)
                peak.append(len(active))
                time.sleep(.03)
                active.pop()
                return super().detect(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "serial.avi"
            self.make_video(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", SerializedRecognizer):
                worker.start("video", path)
                try:
                    worker.frames.get(timeout=3)
                    self.assertTrue(worker.check_frame(np.zeros((100, 220, 3), np.uint8), "video", str(path), .2))
                    worker.frame_worker.thread.join(5)
                    self.assertFalse(worker.frame_worker.alive)
                finally:
                    worker.stop()
                    worker.thread.join(5)
            self.assertEqual(max(peak), 1)

    def make_video(self, path, count=40, fps=20):
        import cv2
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (220, 100))
        self.assertTrue(writer.isOpened())
        for index in range(count):
            writer.write(np.full((100, 220, 3), 20 + index * 3, np.uint8))
        writer.release()

    def test_video_plays_during_slow_model_startup_and_checks_every_sample(self):
        import threading
        loading, release = threading.Event(), threading.Event()
        class SlowRecognizer(FakeRecognizer):
            def __init__(self, *args, **kwargs):
                loading.set()
                release.wait(5)
            def detect(self, frame):
                time.sleep(.08)
                return super().detect(frame)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "Монгол видео.avi"
            self.make_video(path)
            settings = deepcopy(DEFAULTS)
            settings["frame_stride"] = 2
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", SlowRecognizer):
                worker.start("video", path)
                try:
                    self.assertTrue(loading.wait(3))
                    first = worker.frames.get(timeout=2)
                    time.sleep(.4)
                    later = worker.frames.get(timeout=2)
                    self.assertGreater(float(later.mean()), float(first.mean()) + 6)
                    self.assertEqual(worker.processed, 0)
                finally:
                    release.set()
                    worker.thread.join(8)
                    worker.stop_event.set()
            self.assertFalse(worker.alive)
            self.assertEqual(worker.processed, 20)
            self.assertEqual(worker.saved_count, 1)
            self.assertNotIn("error", [kind for kind, _ in worker.events.queue])

    def test_video_model_failure_does_not_stop_playback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "preview.avi"
            self.make_video(path, count=20)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", side_effect=RuntimeError("model unavailable")):
                worker.start("video", path)
                worker.thread.join(5)
            self.assertFalse(worker.alive)
            self.assertGreater(worker.frames.get_nowait().mean(), 65)
            events = list(worker.events.queue)
            self.assertIn("analysis_error", [kind for kind, _ in events])
            self.assertEqual(len([value for kind, value in events if kind == "playback"]), 20)
            self.assertFalse(any(kind == "status" and value.startswith("Дууслаа") for kind, value in events))

    def test_camera_model_failure_keeps_stream_until_explicit_stop(self):
        class Camera:
            released = False
            reads = 0
            def isOpened(self):
                return True
            def read(self):
                time.sleep(.015)
                self.reads += 1
                return True, np.full((100, 220, 3), self.reads % 255, np.uint8)
            def release(self):
                self.released = True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            camera = Camera()
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            with patch("cv2.VideoCapture", return_value=camera), patch("platewatch.worker.PlateRecognizer", side_effect=RuntimeError("model unavailable")):
                worker.start("camera", "auto")
                try:
                    deadline = time.monotonic() + 3
                    while not worker.analysis_failed and time.monotonic() < deadline:
                        time.sleep(.02)
                    reads = camera.reads
                    time.sleep(.15)
                    self.assertTrue(worker.alive)
                    self.assertFalse(worker.stop_event.is_set())
                    self.assertGreater(camera.reads, reads + 2)
                finally:
                    worker.stop_event.set()
                    worker.thread.join(5)
            self.assertTrue(camera.released)
            self.assertFalse(worker.alive)

    def test_image_is_published_before_ai_finishes_loading(self):
        entered, resume = threading.Event(), threading.Event()
        class SlowRecognizer(FakeRecognizer):
            def __init__(self, *args, **kwargs):
                entered.set()
                resume.wait(5)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "preview.png"
            Image.new("RGB", (220, 100), (10, 20, 30)).save(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            with patch("platewatch.worker.PlateRecognizer", SlowRecognizer):
                worker.start("image", [path])
                try:
                    self.assertTrue(entered.wait(3))
                    preview = worker.frames.get_nowait()
                    self.assertEqual(tuple(preview[0, 0]), (10, 20, 30))
                    self.assertFalse(worker.store.records())
                finally:
                    resume.set()
                    worker.thread.join(6)
            self.assertEqual(len(worker.store.records()), 1)

    def test_ai_failure_keeps_selected_image_visible(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "preview.png"
            Image.new("RGB", (220, 100), "white").save(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            with patch("platewatch.worker.PlateRecognizer", side_effect=RuntimeError("model failed")):
                worker.start("image", [path])
                worker.thread.join(6)
            self.assertFalse(worker.frames.empty())
            self.assertIn("error", [kind for kind, _ in list(worker.events.queue)])
    def test_camera_capture_and_stop_release_without_real_hardware(self):
        class FakeCamera:
            def __init__(self):
                self.released = False

            def isOpened(self):
                return True

            def set(self, *args):
                return True

            def read(self):
                time.sleep(.015)
                return True, np.full((100, 220, 3), 200, np.uint8)

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            camera = FakeCamera()
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(Path(temp) / "data"))
            with patch("platewatch.worker.PlateRecognizer", FakeRecognizer), patch("cv2.VideoCapture", return_value=camera):
                worker.start("camera", 0)
                deadline = time.monotonic() + 5
                while not worker.store.records() and time.monotonic() < deadline:
                    time.sleep(.02)
                worker.stop_event.set()
                worker.thread.join(timeout=5)
            self.assertFalse(worker.alive)
            self.assertTrue(camera.released)
            self.assertEqual(len(worker.store.records()), 1)
            self.assertEqual(worker.store.records()[0]["source_kind"], "camera")

    def test_image_batch_unicode_paths_and_repeat_plate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            paths = [root / "Монгол зураг 1.png", root / "Монгол зураг 2.png"]
            for path in paths:
                Image.fromarray(np.full((100, 220, 3), 225, np.uint8)).save(path)
            worker = DetectionWorker(deepcopy(DEFAULTS), RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", FakeRecognizer):
                worker.start("image", paths)
                worker.thread.join(timeout=10)
            self.assertFalse(worker.alive)
            self.assertEqual(len(worker.store.records()), 2)
            events = list(worker.events.queue)
            self.assertNotIn("error", [k for k, v in events])
            self.assertIn("done", [k for k, v in events])

    def test_video_sampling_confirmation_and_output_file(self):
        import cv2
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "sample.avi"
            video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (220, 100))
            self.assertTrue(video.isOpened())
            for _ in range(12):
                video.write(np.full((100, 220, 3), 200, np.uint8))
            video.release()
            settings = deepcopy(DEFAULTS)
            settings.update(frame_stride=2, save_video=True, adaptive_sampling=False)
            worker = DetectionWorker(settings, RecordStore(root / "data"))
            with patch("platewatch.worker.PlateRecognizer", FakeRecognizer):
                worker.start("video", path)
                worker.thread.join(timeout=15)
            self.assertFalse(worker.alive)
            self.assertEqual(len(worker.store.records()), 1)
            self.assertEqual(float(worker.store.records()[0]["video_seconds"]), .2)
            self.assertTrue(list((root / "data" / "videos").glob("*.mp4")))
            self.assertNotIn("error", [k for k, v in list(worker.events.queue)])


if __name__ == "__main__":
    unittest.main()
