r"""Run an actual YOLO/OCR image through the GUI, keeping evidence in test_artifacts.

Usage: .venv\Scripts\python.exe -m tools.verify_gui --image test_artifacts\synthetic_car.png
The image must contain a readable Mongolian 4+3 plate. The window closes automatically.
"""
import argparse
from pathlib import Path
import time
from PIL import ImageGrab
from platewatch.gui import PlateWatchApp
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--screenshot", action="store_true")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error("Image not found")
    app = PlateWatchApp()
    app.store = RecordStore(Path("test_artifacts") / "gui_evidence")
    existing = {record["id"] for record in app.store.records()}
    app.worker = DetectionWorker(app.settings, app.store)
    app.refresh_records()
    app.start_source("image", [str(args.image.resolve())])
    started = time.monotonic()
    outcome = {"ok": False}

    def inspect():
        new_records = [record for record in app.store.records() if record["id"] not in existing]
        if not app.worker.alive and new_records:
            app.refresh_records()
            outcome["ok"] = True
            def finish():
                if args.screenshot:
                    ImageGrab.grab(window=app.winfo_id()).save("test_artifacts/gui_detection.png")
                print("GUI PIPELINE OK:", app.store.records()[-1]["plate"], flush=True)
                app.show_page("rules")
                app.rule_plate.set("1234УБА")
                app.rule_date.set("2026-11-12")
                app.rule_time.set("12:00")
                app.check_rule()
                assert app.last_rule_result.code == "restricted"
                print("RULE PAGE OK", flush=True)
                app.show_page("history")
                assert app.history_table.get_children()
                print("HISTORY PAGE OK", flush=True)
                app.close()
            app.after(300, finish)
        elif time.monotonic() - started > 90 or (not app.worker.alive and time.monotonic() - started > 8):
            print("GUI PIPELINE FAILED:", app.status_text.get(), flush=True)
            app.close()
        else:
            app.after(250, inspect)
    app.after(250, inspect)
    app.mainloop()
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
