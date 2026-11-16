"""Exercise the reported real video using the actual GUI, detector and OCR."""
import argparse
from pathlib import Path
import time
from PIL import ImageGrab
from platewatch.gui import PlateWatchApp
from platewatch.settings import atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    folder = Path("test_artifacts/accuracy_speed_2_6")
    folder.mkdir(exist_ok=True)
    app = PlateWatchApp()
    app.settings.update(frame_stride=10, recognition_mode="balanced", video_min_confidence=.65)
    app.store = RecordStore(folder / "data")
    existing = {r["id"] for r in app.store.records()}
    app.worker = DetectionWorker(app.settings, app.store)
    app.video_stride_var.set("10")
    app.recognition_mode_var.set("Тэнцвэртэй")
    app.refresh_records()
    report = {"ok": False, "source": str(args.video.resolve()), "version": app.title(),
              "stride": 10, "mode": "balanced"}
    started = time.monotonic()
    captured = False

    def inspect():
        nonlocal captured
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if time.monotonic() - started > 180:
                raise TimeoutError(app.status_text.get())
            if "7034УБУ" in app.session_records and not captured:
                app.update_idletasks()
                ImageGrab.grab(window=app.winfo_id()).save(folder / "confirmed_plate.png")
                report["first_plate"] = app.plate_label.cget("text")
                report["first_plate_elapsed_seconds"] = round(time.monotonic() - started, 2)
                captured = True
            if app.worker.processed and not app.worker.alive:
                records = [r for r in app.store.records() if r["id"] not in existing]
                report.update(elapsed_seconds=round(time.monotonic() - started, 2),
                              processed_frames=app.worker.processed,
                              records=[{k: r[k] for k in ("plate", "video_seconds", "ocr_confidence", "crop_path")} for r in records])
                target_records = [r["plate"] for r in records if float(r["video_seconds"]) < 1.7]
                report["target_records"] = target_records
                report["ok"] = (target_records == ["7034УБУ"] and app.progress.get() == 1
                                and app.preview_displayed and not app.worker.analysis_failed)
                app.close()
                return
        except Exception as exc:
            report["error"] = str(exc)
            app.close()
            return
        app.after(80, inspect)

    app.after(200, lambda: app.start_source("video", str(args.video.resolve()), review=True))
    app.after(300, inspect)
    app.mainloop()
    atomic_json(folder / "report.json", report)
    print(report, flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
