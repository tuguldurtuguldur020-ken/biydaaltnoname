"""Exercise the actual Tk button callbacks, including source changes and failures."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import tempfile
import time
import logging
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from platewatch.gui import PlateWatchApp
from platewatch.plates import Detection
from platewatch.rules import evaluate
from platewatch.settings import DEFAULTS, UB_TIME
from platewatch.storage import RecordStore


class Recognizer:
    def __init__(self, *args, **kwargs):
        pass
    def detect(self, frame):
        return [Detection((10, 10, 140, 60), "1234УБА", .95, .9, "1234УБА")]


class GuiTests(unittest.TestCase):
    def test_settings_reuse_loaded_models_and_preserve_rotation(self):
        self.app.start_source("image", [str(self.path)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        recognizer = self.app.worker.recognizer
        self.app.worker.settings["_rotation"] = 270
        self.app.video_stride_var.set("17")
        self.assertTrue(self.app.apply_settings())
        self.assertIs(self.app.worker.recognizer, recognizer)
        self.assertEqual(self.app.worker.settings["_rotation"], 270)
        self.app.device_var.set("auto")
        self.assertTrue(self.app.apply_settings())
        self.assertIsNone(self.app.worker.recognizer)

    def test_unpadded_date_filter_does_not_match_a_different_day_prefix(self):
        pixels = np.full((100, 220, 3), 200, np.uint8)
        for day in (1, 10):
            at = datetime(2026, 11, day, 12, tzinfo=UB_TIME)
            self.store.add({"plate": "1234УБА", "source": str(self.path), "source_kind": "image",
                            "ocr_confidence": .9, "detector_confidence": .9}, pixels, pixels[:30],
                           evaluate("1234УБА", DEFAULTS["rules"], at), at)
        self.app.query_date.set("2026-11-1")
        self.app.refresh_records()
        self.assertEqual(len(self.app.filtered_records), 1)
        self.assertTrue(self.app.filtered_records[0]["detected_at"].startswith("2026-11-01"))

    def test_batch_manual_check_keeps_the_source_of_the_displayed_image(self):
        second = self.root / "second.png"
        Image.new("RGB", (220, 100), (40, 50, 60)).save(second)
        self.app.start_source("image", [str(self.path), str(second)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive and self.app._frame_source == str(second))
        self.app.frame_button.invoke()
        self.pump(lambda: not self.app.frame_check_busy and self.app._frame_pinned)
        self.assertEqual(self.store.records()[-1]["source"], str(second))
        self.assertEqual(tuple(self.app._frame_original[0, 0]), (40, 50, 60))

    def test_manual_video_check_uses_the_displayed_frame_timestamp(self):
        self.app.active_kind, self.app.active_source = "video", "clip.mp4"
        self.app.worker.publish(np.full((100, 220, 3), 100, np.uint8), seconds=2 / 30)
        self.app._poll_once()
        self.app.playback_seconds = 3 / 30
        with patch.object(self.app.worker, "check_frame", return_value=False) as check:
            self.app.check_current_frame()
        self.assertAlmostEqual(check.call_args.args[3], 2 / 30)

    def setUp(self):
        self.log_handlers = set(logging.getLogger().handlers)
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)
        self.store = RecordStore(self.root / "data")
        self.path = self.root / "Монгол зураг.png"
        Image.new("RGB", (220, 100), (15, 30, 45)).save(self.path)
        self.patches = [patch("platewatch.gui.RecordStore", return_value=self.store),
                        patch("platewatch.gui.DATA_DIR", self.root / "gui"),
                        patch("platewatch.gui.load_settings", return_value=deepcopy(DEFAULTS)),
                        patch("platewatch.gui.save_settings"),
                        patch("platewatch.worker.PlateRecognizer", Recognizer)]
        for mock in self.patches:
            mock.start()
        self.app = PlateWatchApp()
        self.app.withdraw()

    def tearDown(self):
        self.app.worker.stop_event.set()
        self.app.worker.pause_event.clear()
        if self.app.worker.thread:
            self.app.worker.thread.join(5)
        self.app.close()
        for mock in reversed(self.patches):
            mock.stop()
        for handler in set(logging.getLogger().handlers) - self.log_handlers:
            logging.getLogger().removeHandler(handler)
            handler.close()
        self.temp.cleanup()

    def pump(self, condition, timeout=7):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.update()
            if condition():
                return
            time.sleep(.02)
        self.fail("GUI condition timed out: " + self.app.status_text.get())

    def test_import_button_runs_picker_and_displays_selected_image(self):
        with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)) as picker:
            self.app.image_button.invoke()
            picker.assert_called_once()
        self.pump(lambda: bool(self.app.history_table.get_children()))
        self.assertIn("1234", self.app.plate_label.cget("text"))
        self.assertEqual(self.app.last_rgb.shape[:2], (100, 220))
        self.assertEqual(self.store.records()[0]["source"], str(self.path))

    def test_main_preview_image_button_imports_photo(self):
        with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)) as picker:
            self.app.main_image_button.invoke()
            picker.assert_called_once()
        self.pump(lambda: bool(self.app.history_table.get_children()))
        self.assertIn("1234", self.app.plate_label.cget("text"))

    def test_frozen_frame_button_checks_original_and_keeps_result_pinned(self):
        self.app.start_source("image", [str(self.path)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.app.frame_button.invoke()
        self.pump(lambda: not self.app.frame_check_busy and self.app._frame_pinned)
        self.assertTrue(self.app.worker.pause_event.is_set())
        self.assertIn("1234", self.app.plate_label.cget("text"))
        self.assertEqual(tuple(self.app._frame_original[0, 0]), (15, 30, 45))
        self.assertEqual(self.app.session_list.size(), 1)
        self.assertEqual(len(self.store.records()), 2)
        result = self.app.last_rgb
        self.app.worker.publish(np.zeros_like(result))
        self.app._poll_once()
        self.assertIs(self.app.last_rgb, result)
        self.assertFalse(self.app.frame_check_busy)

    def test_review_button_checks_all_frames_and_lists_results(self):
        import cv2
        path = self.root / "all.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (220, 100))
        self.assertTrue(writer.isOpened())
        for _ in range(8):
            writer.write(np.full((100, 220, 3), 100, np.uint8))
        writer.release()
        self.app.start_source("video", str(path))
        self.pump(lambda: self.app.preview_displayed)
        self.app.video_stride_var.set("1")
        self.app.review_button.invoke()
        self.pump(lambda: self.app.worker.video_review and not self.app.worker.alive
                  and self.app.progress.get() == 1 and self.app.preview_displayed)
        self.assertEqual(self.app.worker.processed, 8)
        self.assertEqual(self.app.session_list.size(), 1)
        self.assertIn("1234", self.app.session_list.get(0))
        self.assertIsNotNone(self.app._frame_original)

    def test_video_controls_apply_live_persist_and_reject_invalid_input(self):
        import cv2
        path = self.root / "controls.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (220, 100))
        self.assertTrue(writer.isOpened())
        for _ in range(40):
            writer.write(np.full((100, 220, 3), 100, np.uint8))
        writer.release()
        self.app.start_source("video", str(path))
        self.pump(lambda: self.app.preview_displayed and self.app.worker.recognizer is not None)
        self.assertTrue(self.app.worker.alive)
        worker, recognizer = self.app.worker, self.app.worker.recognizer
        self.app.recognition_mode_var.set("Хурдан")
        self.app.video_stride_var.set("30")
        self.app.playback_speed_var.set("2.0")
        self.assertTrue(self.app.apply_video_controls())
        self.assertIs(self.app.worker, worker)
        self.assertIs(worker.recognizer, recognizer)
        self.assertEqual(worker.settings["frame_stride"], 30)
        self.assertEqual(worker.settings["recognition_mode"], "fast")
        self.assertEqual(self.app.settings["playback_speed"], 2)
        self.assertEqual(self.app.setting_vars["frame_stride"].get(), "30")
        self.app.video_stride_var.set("0")
        self.assertFalse(self.app.apply_video_controls())
        self.assertEqual(worker.settings["frame_stride"], 30)


    def test_preview_upscales_and_fullscreen_escape_restores_without_stopping(self):
        self.app.deiconify()
        self.app.update()
        self.app.last_rgb = np.full((100, 220, 3), 100, np.uint8)
        self.app.redraw_preview()
        normal_width = self.app.preview.winfo_width()
        self.assertGreater(self.app.preview_photo.width(), 220)
        self.app.toggle_fullscreen()
        self.app.update()
        self.app.redraw_preview()
        self.assertTrue(self.app.fullscreen)
        self.assertGreater(self.app.preview.winfo_width(), normal_width)
        self.assertFalse(self.app.sidebar.winfo_ismapped())
        with patch.object(self.app, "stop") as stop:
            self.app.escape()
            stop.assert_not_called()
        self.app.update()
        self.assertFalse(self.app.fullscreen)
        self.assertTrue(self.app.sidebar.winfo_ismapped())

    def test_fill_preview_covers_area_without_changing_original_frame(self):
        self.app.deiconify()
        self.app.update()
        frame = np.full((500, 200, 3), 100, np.uint8)
        self.app.last_rgb = frame
        self.app.toggle_fill()
        width, height = self.app.preview.winfo_width(), self.app.preview.winfo_height()
        self.assertAlmostEqual(self.app.preview_photo.width() / self.app.preview_photo.height(), width / height, delta=.03)
        self.assertIs(self.app.last_rgb, frame)

    def test_video_picker_rejects_photo_and_preserves_current_source(self):
        with patch("platewatch.gui.filedialog.askopenfilename", return_value=str(self.path)):
            self.app.video_button.invoke()
        self.assertIn("зургийн файл", self.app.last_error)
        self.assertFalse(self.app.worker.alive)

    def test_rotation_rechecks_the_source_and_new_import_resets_rotation(self):
        self.app.start_source("image", [str(self.path)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.app.rotate_source()
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.assertEqual(self.app.last_rgb.shape[:2], (220, 100))
        another = self.root / "another.png"
        Image.new("RGB", (300, 150), "white").save(another)
        self.app.start_source("image", [str(another)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.assertEqual(self.app.last_rgb.shape[:2], (150, 300))

    def test_accent_change_rebuilds_canvas_and_keeps_imported_image_visible(self):
        self.app.start_source("image", [str(self.path)])
        self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.app.accent_var.set("Ногоон")
        self.assertTrue(self.app.apply_settings())
        self.app.redraw_preview()
        self.assertEqual(self.app.preview.type(self.app._preview_item), "image")
        self.assertTrue(self.app.preview.itemcget(self.app._preview_item, "image"))
        self.assertEqual(self.app.last_rgb.shape[:2], (100, 220))


    def test_imported_photo_retries_preview_after_memory_error_without_new_frame(self):
        render = self.app._render_preview
        failures = []
        def fail_once():
            if self.app.last_rgb is not None and not failures:
                failures.append(True)
                raise MemoryError("Simulated preview allocation failure")
            render()
        with patch.object(self.app, "_render_preview", side_effect=fail_once):
            with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)):
                self.app.main_image_button.invoke()
            self.pump(lambda: self.app.preview_displayed and not self.app.worker.alive)
        self.assertEqual(len(failures), 1)
        self.assertIsNotNone(self.app.preview_photo)
        self.assertFalse(self.app._preview_failed)
        self.assertFalse(self.app._preview_dirty)
        self.assertTrue(self.app.worker.frames.empty())
        self.assertTrue(self.app.history_table.get_children())

    def test_poll_reschedules_after_unexpected_callback_failure(self):
        poll = self.app._poll_once
        attempts = []
        def fail_once():
            attempts.append(True)
            if len(attempts) == 1:
                raise RuntimeError("Simulated GUI callback failure")
            poll()
        with patch.object(self.app, "_poll_once", side_effect=fail_once):
            self.app.start_source("image", [str(self.path)])
            self.pump(lambda: self.app.preview_displayed and bool(self.app.history_table.get_children()))
        self.assertGreaterEqual(len(attempts), 2)

    def test_camera_reuses_tk_image_and_import_still_works_after_render_failure(self):
        class Camera:
            released = False
            def isOpened(self):
                return True
            def read(self):
                time.sleep(.015)
                return True, np.full((90, 160, 3), 200, np.uint8)
            def release(self):
                self.released = True
        camera = Camera()
        with patch("cv2.VideoCapture", return_value=camera):
            self.app.camera_button.invoke()
            self.pump(lambda: self.app.preview_displayed)
            photo = self.app.preview_photo
            with patch.object(photo, "paste", wraps=photo.paste) as paste:
                self.pump(lambda: paste.call_count >= 3)
            self.assertIs(self.app.preview_photo, photo)
            with patch.object(self.app, "_render_preview", side_effect=MemoryError("Camera allocation failed")):
                self.app._preview_retry_at = 0
                self.assertFalse(self.app.redraw_preview())
            self.pump(lambda: not self.app._preview_failed)
            with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)):
                self.app.main_image_button.invoke()
            self.pump(lambda: self.app.active_kind == "image" and self.app.preview_displayed
                      and any(r["source_kind"] == "image" for r in self.store.records()))
        self.assertTrue(camera.released)
        self.assertEqual(self.app.last_rgb.shape[:2], (100, 220))

    def test_large_photo_preview_is_bounded_and_keeps_original_frame(self):
        frame = np.full((2100, 3200, 3), 150, np.uint8)
        self.app.worker.publish(frame)
        self.pump(lambda: self.app.preview_displayed)
        self.assertIs(self.app.last_rgb, frame)
        self.assertLessEqual(max(self.app.preview_photo.width(), self.app.preview_photo.height()), 960)

    def test_1113_shows_wednesday_separately_from_selected_friday(self):
        self.app.nav["rules"].invoke()
        self.app.rule_plate.set("1113")
        self.assertIn("Лхагва", self.app.rule_result_label.cget("text"))
        self.assertNotIn("Баасан", self.app.rule_result_label.cget("text"))
        self.app.rule_date.set("2026-11-13")
        self.app.rule_time.set("12:00")
        self.app.check_rule()
        self.assertIn("Лхагва", self.app.rule_result_label.cget("text"))
        self.assertIn("Баасан", self.app.rule_date_result.cget("text"))
        self.assertEqual(self.app.last_rule_result.code, "allowed")
        self.app.rule_date.set("2026-11-11")
        self.app.check_rule()
        self.assertEqual(self.app.last_rule_result.code, "restricted")

    def test_incomplete_csv_row_does_not_break_history_or_rule_lookup(self):
        at = datetime(2026, 11, 13, 11, 39, tzinfo=UB_TIME)
        frame = np.full((100, 220, 3), 200, np.uint8)
        self.store.add({"plate": "6775УБА", "source": str(self.path), "source_kind": "image",
                        "ocr_confidence": .9, "detector_confidence": .95}, frame, frame[:30],
                       evaluate("6775УБА", self.app.settings["rules"], at), at)
        with self.store.csv_path.open("a", encoding="utf-8", newline="") as file:
            file.write("log\n")

        self.app.show_page("history")
        self.app.clear_filters()
        self.assertEqual(len(self.app.history_table.get_children()), 1)
        self.app.show_page("rules")
        self.app.rule_plate.set("6775")
        self.app.rule_date.set("2026-11-13")
        self.app.rule_time.set("11:39")
        self.app.check_rule()
        self.assertEqual(self.app.last_rule_result.code, "restricted")
        self.assertIn("1 бүртгэл", self.app.rule_seen_label.cget("text"))

    def test_import_cancellation_does_not_start_inference(self):
        with patch("platewatch.gui.filedialog.askopenfilenames", return_value=()):
            self.app.image_button.invoke()
        self.assertFalse(self.app.worker.alive)
        self.assertIsNone(self.app.pending_source)

    def test_video_import_button_processes_selected_file(self):
        import cv2
        path = self.root / "Видео бичлэг.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (220, 100))
        self.assertTrue(writer.isOpened())
        for _ in range(12):
            writer.write(np.full((100, 220, 3), 150, np.uint8))
        writer.release()
        self.app.video_stride_var.set("2")
        self.assertTrue(self.app.apply_video_controls())
        with patch("platewatch.gui.filedialog.askopenfilename", return_value=str(path)) as picker:
            self.app.video_button.invoke()
            picker.assert_called_once()
        self.pump(lambda: any(r["source_kind"] == "video" for r in self.store.records()))
        self.assertEqual(self.app.active_kind, "video")

    def test_file_path_import_and_error_feedback(self):
        self.app.import_path_var.set(str(self.path))
        self.app.open_import_path()
        self.pump(lambda: bool(self.store.records()))
        self.app.import_path_var.set(str(self.root / "missing.png"))
        self.app.open_import_path()
        self.assertIn("Файл олдсонгүй", self.app.last_error)
        self.app.nav["rules"].invoke()
        self.assertEqual(self.app.current_page, "rules")

    def test_camera_to_image_switch_releases_device_and_imports(self):
        class Camera:
            released = False
            def isOpened(self):
                return True
            def read(self):
                time.sleep(.015)
                return True, np.full((90, 160, 3), 200, np.uint8)
            def release(self):
                self.released = True
        camera = Camera()
        with patch("cv2.VideoCapture", return_value=camera):
            self.app.camera_button.invoke()
            self.pump(lambda: self.app.last_rgb is not None and bool(self.store.records()))
            with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)):
                self.app.image_button.invoke()
            self.pump(lambda: any(r["source_kind"] == "image" for r in self.store.records()))
        self.pump(lambda: self.app.last_rgb is not None and self.app.last_rgb.shape[:2] == (100, 220))
        self.assertTrue(camera.released)
        self.assertIsNone(self.app.pending_source)

    def test_image_stays_visible_and_gui_remains_responsive_on_ai_error(self):
        with patch("platewatch.worker.PlateRecognizer", side_effect=RuntimeError("AI test failure")):
            with patch("platewatch.gui.filedialog.askopenfilenames", return_value=(str(self.path),)):
                self.app.image_button.invoke()
            self.pump(lambda: bool(self.app.last_error))
        self.assertIsNotNone(self.app.last_rgb)
        self.assertIn("AI test failure", self.app.last_error)
        self.app.nav["settings"].invoke()
        self.assertEqual(self.app.current_page, "settings")
