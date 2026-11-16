"""Exercise actual video playback, YOLO/OCR, fullscreen and an optional camera.

Uses an isolated evidence directory and the installed models. Run checks one at
a time on small PCs. Ordinary camera frames are not written by this verifier.
"""
import argparse
import json
from pathlib import Path
import time

from PIL import ImageGrab
from platewatch.gui import PlateWatchApp
from platewatch.media import open_camera
from platewatch.settings import atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--report", type=Path, default=Path("test_artifacts/media_verification_2_4.json"))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error("Video not found")
    import cv2
    from platewatch.media import read_image
    fixture = Path("test_artifacts/Монгол_дугаар_туршилт.mp4").resolve()
    frame = read_image("test_artifacts/synthetic_car.png")
    writer = cv2.VideoWriter(str(fixture), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                             (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        parser.error("Could not create fixture video")
    for _ in range(30):
        writer.write(frame)
    writer.release()

    app = PlateWatchApp()
    if app.tk.call("tk", "windowingsystem") == "win32":
        # Automated pause/replay checks must not receive concurrent keyboard or
        # mouse input intended for the user's other windows.
        app.attributes("-disabled", True)
    app.store = RecordStore(Path("test_artifacts/media_evidence_2_4"))
    app.worker = DetectionWorker(app.settings, app.store)
    app.refresh_records()
    report = {"ok": False, "version": app.title(), "video": str(args.video.resolve())}
    phase = "video"
    phase_started = time.monotonic()
    rendered = []
    render = app._render_preview
    pause_at = None
    paused_position = None
    phase_record_offset = len(app.store.records())

    def tracked_render():
        render()
        if app.last_rgb is not None:
            rendered.append(time.monotonic())
    app._render_preview = tracked_render

    def begin(kind, source, next_phase):
        nonlocal phase, phase_started, phase_record_offset
        phase, phase_started = next_phase, time.monotonic()
        phase_record_offset = len(app.store.records())
        rendered.clear()
        print(f"Checking {next_phase}", flush=True)
        app.start_source(kind, source, args.rotation if next_phase == "video" else 0)

    def capture(name):
        app.update()
        time.sleep(.15)
        app.update()
        ImageGrab.grab(window=app.winfo_id()).save(str(Path("test_artifacts") / name))

    def finish():
        report["ok"] = bool(report.get("video_ok") and report.get("fixture_ok")
                            and report.get("pause_ok") and report.get("fullscreen_ok")
                            and (not args.camera or report.get("camera_ok")))
        app.close()

    def inspect():
        nonlocal pause_at, paused_position
        elapsed = time.monotonic() - phase_started
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if elapsed > 160:
                raise TimeoutError(f"{phase} timed out: {app.status_text.get()}")
            if phase == "video":
                if app.preview_displayed:
                    report.setdefault("video_first_preview_seconds", round(elapsed, 3))
                if app.playback_seconds >= 1 and pause_at is None:
                    app.pause()
                    pause_at = time.monotonic()
                if pause_at is not None and not report.get("pause_checked"):
                    if time.monotonic() - pause_at > .15 and paused_position is None:
                        paused_position = app.playback_seconds
                    if time.monotonic() - pause_at > .5:
                        report["pause_ok"] = abs(app.playback_seconds - paused_position) < .1
                        report["pause_checked"] = True
                        app.pause()
                if app.playback_seconds >= 2 and not report.get("fullscreen_ok"):
                    normal_width = app.preview.winfo_width()
                    app.toggle_fullscreen()
                    app.update_idletasks()
                    app.redraw_preview()
                    report["fullscreen_ok"] = app.fullscreen and app.preview.winfo_width() > normal_width
                    capture("video_fullscreen_2_4.png")
                    app.escape()
                    report["fullscreen_escape_keeps_playing"] = not app.worker.stop_event.is_set()
                    capture("video_monitor_2_4.png")
                if not app.worker.alive and app.progress.get() == 1 and elapsed > 1:
                    report["video_processed_samples"] = app.worker.processed
                    report["video_duration"] = app.video_duration
                    report["video_playback_seconds"] = app.playback_seconds
                    report["video_rendered_frames"] = len(rendered)
                    report["video_plate_records"] = app.worker.saved_count
                    report["video_plates"] = [r["plate"] for r in app.store.records()[phase_record_offset:]]
                    report["video_rotation"] = args.rotation
                    report["video_ok"] = (app.preview_displayed and app.progress.get() == 1
                                          and app.playback_seconds >= app.video_duration - .2
                                          and app.worker.processed > 0)
                    begin("video", fixture, "fixture")
            elif phase == "fixture" and not app.worker.alive and app.progress.get() == 1 and elapsed > 1:
                records = [r for r in app.store.records()[phase_record_offset:] if r["source"] == str(fixture)]
                report["fixture_plates"] = [r["plate"] for r in records]
                report["fixture_processed_samples"] = app.worker.processed
                report["fixture_ok"] = any(r["plate"] == "1234УБА" for r in records) and app.worker.processed == 6 and app.worker.saved_count > 0
                capture("video_recognition_2_4.png")
                if args.camera:
                    begin("camera", "auto", "camera")
                else:
                    finish()
                    return
            elif phase == "camera" and app.preview_displayed and app.worker.processed >= 3:
                report["camera_preview_frames"] = len(rendered)
                report["camera_inference_frames"] = app.worker.processed
                report["camera_shape"] = list(app.last_rgb.shape)
                report["camera_index"] = app.worker.actual_camera_index
                app.stop()
                phase_change("camera_stopping")
            elif phase == "camera_stopping" and not app.worker.alive:
                capture_device, camera_frame, backend, index = open_camera(report["camera_index"])
                report["camera_reopened_after_stop"] = bool(camera_frame.size)
                capture_device.release()
                report["camera_ok"] = report["camera_reopened_after_stop"] and report["camera_inference_frames"] >= 3
                finish()
                return
        except Exception as exc:
            report["error"] = str(exc)
            app.close()
            return
        app.after(80, inspect)

    def phase_change(value):
        nonlocal phase, phase_started
        phase, phase_started = value, time.monotonic()

    app.after(200, lambda: begin("video", str(args.video.resolve()), "video"))
    app.after(300, inspect)
    app.mainloop()
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
