"""Verify a paused downloaded video and detailed review using actual YOLO/OCR."""
import argparse
import json
from pathlib import Path
import time
from PIL import ImageGrab
from platewatch.gui import PlateWatchApp
from platewatch.media import open_video
from platewatch.settings import atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=11.0)
    parser.add_argument("--report", type=Path, default=Path("test_artifacts/review_verification_2_5.json"))
    args = parser.parse_args()
    import cv2
    source = str(args.video.resolve())
    capture, frame, info = open_video(source)
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(args.seconds * info["fps"]))
    clip = Path("test_artifacts/Зогсоосон_кадр_туршилт.mp4").resolve()
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (frame.shape[1], frame.shape[0]))
    clip_frames = 0
    for _ in range(12):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        clip_frames += 1
    capture.release()
    writer.release()

    app = PlateWatchApp()
    if app.tk.call("tk", "windowingsystem") == "win32":
        app.attributes("-disabled", True)
    app.store = RecordStore(Path("test_artifacts/review_evidence_2_5"))
    app.worker = DetectionWorker(app.settings, app.store)
    app.refresh_records()
    report = {"ok": False, "version": app.title(), "source": source, "target_seconds": args.seconds}
    phase = "playback"
    started = time.monotonic()
    original = None

    def screenshot(name):
        app.update()
        time.sleep(.15)
        app.update()
        ImageGrab.grab(window=app.winfo_id()).save(str(Path("test_artifacts") / name))

    def inspect():
        nonlocal phase, started, original
        elapsed = time.monotonic() - started
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if elapsed > 180:
                raise TimeoutError(phase + ": " + app.status_text.get())
            if phase == "playback" and app.preview_displayed and app.playback_seconds >= args.seconds:
                original = app.last_rgb.copy()
                report["paused_seconds"] = app.playback_seconds
                app.frame_button.invoke()
                report["frame_button_started"] = app.frame_check_busy
                phase, started = "manual", time.monotonic()
                print("Checking frozen frame", flush=True)
            elif phase == "manual" and not app.frame_check_busy and app._frame_pinned:
                import numpy as np
                report["exact_original_preserved"] = bool(np.array_equal(original, app._requested_original))
                report["recognized_frame_seconds"] = app.playback_seconds
                report["manual_model_shared"] = app.worker.recognizer is app.worker.frame_worker.recognizer
                report["manual_saved"] = app.worker.frame_worker.saved_count
                report["manual_display_plate"] = app.plate_label.cget("text")
                report["manual_raw_text"] = app.last_detection.raw_text if app.last_detection else ""
                report["paused_preview_shape"] = list(app._frame_original.shape)
                screenshot("paused_frame_2_5.png")
                app.stop()
                phase, started = "stopping", time.monotonic()
            elif phase == "stopping" and not app.worker.alive:
                app.start_source("video", str(clip), review=True)
                phase, started = "review", time.monotonic()
                print("Checking every frame of the original-video excerpt", flush=True)
            elif phase == "review" and not app.worker.alive and app.progress.get() == 1:
                report["review_expected_frames"] = clip_frames
                report["review_processed_frames"] = app.worker.processed
                report["review_plates"] = list(app.session_records)
                report["review_last_frame_seconds"] = app.playback_seconds
                screenshot("detailed_review_2_5.png")
                report["ok"] = bool(report["exact_original_preserved"] and report["manual_model_shared"]
                                    and app.worker.processed == clip_frames and app.preview_displayed)
                app.close()
                return
        except Exception as exc:
            report["error"] = str(exc)
            app.close()
            return
        app.after(80, inspect)

    app.after(200, lambda: app.start_source("video", source))
    app.after(300, inspect)
    app.mainloop()
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
