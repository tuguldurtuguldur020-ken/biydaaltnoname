"""Actual GUI/AI regression for later cars and a partially readable frozen frame."""
import argparse
from pathlib import Path
import time

from PIL import ImageGrab

from platewatch.gui import PlateWatchApp
from platewatch import gui
from platewatch.settings import atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mode", choices=["balanced", "detailed"], default="detailed")
    args = parser.parse_args()
    folder = Path("test_artifacts/accuracy_2_7")
    gui.DATA_DIR = folder / "gui"
    source = str(args.video.resolve())
    app = PlateWatchApp()
    if app.tk.call("tk", "windowingsystem") == "win32":
        app.attributes("-disabled", True)
    app.settings.update(frame_stride=10, recognition_mode=args.mode, neighbor_frames=2)
    app.store = RecordStore(folder / "evidence")
    before = {r["id"] for r in app.store.records()}
    app.worker = DetectionWorker(app.settings, app.store)
    app.video_stride_var.set("10")
    app.recognition_mode_var.set("Нарийвчилсан" if args.mode == "detailed" else "Тэнцвэртэй")
    app.refresh_records()
    expected = {"7034УБУ", "6657УНА", "4000УЕО"}
    report = {"ok": False, "version": app.title(), "mode": args.mode}
    started = time.monotonic()
    phase = "video"
    screenshot = False

    def inspect():
        nonlocal phase, screenshot
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if time.monotonic() - started > 180:
                raise TimeoutError(app.status_text.get())
            if phase == "video":
                if "6657УНА" in app.session_records and not screenshot:
                    app.update_idletasks()
                    ImageGrab.grab(window=app.winfo_id()).save(folder / (args.mode + "_corrected.png"))
                    screenshot = True
                if app.worker.processed and not app.worker.alive:
                    if app.progress.get() < 1 or app.worker.analysis_failed:
                        raise RuntimeError("Video verification stopped before completion: " + app.status_text.get())
                    records = [r for r in app.store.records() if r["id"] not in before]
                    report.update(video_elapsed_seconds=round(time.monotonic() - started, 2),
                                  processed_frames=app.worker.processed,
                                  video_plates=[r["plate"] for r in records],
                                  expected_plates=sorted(expected))
                    capture = cv2.VideoCapture(source)
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 106)
                    ok, frame = capture.read()
                    capture.release()
                    if not ok:
                        raise RuntimeError("Frozen-frame fixture unavailable")
                    app.last_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    app._frame_original = app.last_rgb
                    app.playback_seconds = 106 / 30
                    app._frame_seconds = app.playback_seconds
                    app.check_current_frame()
                    phase = "manual"
            elif not app.frame_check_busy and app._frame_pinned:
                report["manual_plate"] = app.plate_label.cget("text")
                report["manual_seconds"] = app.playback_seconds
                report["manual_status"] = app.status_text.get()
                app.update_idletasks()
                ImageGrab.grab(window=app.winfo_id()).save(folder / (args.mode + "_frozen.png"))
                saved = set(report["video_plates"])
                report["ok"] = (expected <= saved and not {"6657УНХ", "6657УНЬ", "1703УБУ"} & saved
                                and report["manual_plate"] == "6657 УНА")
                app.close()
                return
        except Exception as exc:
            report["error"] = str(exc)
            app.close()
            return
        app.after(80, inspect)

    app.after(200, lambda: app.start_source("video", source, review=True))
    app.after(300, inspect)
    app.mainloop()
    atomic_json(folder / (args.mode + "_report.json"), report)
    print(report, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
