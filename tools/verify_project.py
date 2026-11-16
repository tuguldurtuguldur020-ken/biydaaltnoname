"""Reproducible real-model and GUI acceptance checks, isolated from user records.

Example: python -m tools.verify_project --manifest cases.json --video clip.mp4 --camera
Manifest: [{"source": "path/to/photo.jpg", "plate": "1742УБР"}, ...]
"""
import argparse
from copy import deepcopy
import csv
import json
import logging
from pathlib import Path
import queue
import time

from platewatch import __version__
from platewatch.media import open_camera
from platewatch.recognition import PlateRecognizer
from platewatch.settings import DEFAULTS, atomic_json, load_settings, save_settings
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def worker_result(worker, timeout=120):
    deadline = time.monotonic() + timeout
    while worker.alive and time.monotonic() < deadline:
        worker.thread.join(.1)
    if worker.alive:
        worker.stop()
        worker.thread.join(10)
        raise TimeoutError("Worker did not complete")
    events = list(worker.events.queue)
    errors = [value for kind, value in events if kind in ("error", "analysis_error", "frame_error")]
    if errors:
        raise RuntimeError("; ".join(errors))
    return [value for kind, value in events if kind == "record"]


def verify_gui(settings, recognizer, store, case, folder):
    from PIL import ImageGrab
    from platewatch import gui
    gui.DATA_DIR = folder / "gui_state"
    gui.load_settings = lambda: deepcopy(settings)
    gui.save_settings = lambda values: save_settings(values, gui.DATA_DIR)
    app = gui.PlateWatchApp()
    if app.tk.call("tk", "windowingsystem") == "win32":
        app.attributes("-disabled", True)
    app.store = store
    app.worker = DetectionWorker(settings, store)
    app.worker.recognizer = recognizer
    app.refresh_records()
    result = {"ok": False}
    phase = "image"
    started = time.monotonic()

    def inspect():
        nonlocal phase
        try:
            if app.last_error:
                raise RuntimeError(app.last_error)
            if time.monotonic() - started > 120:
                raise TimeoutError(app.status_text.get())
            if phase == "image" and not app.worker.alive and app.preview_displayed:
                result["plate"] = app.plate_label.cget("text").replace(" ", "")
                result["preview"] = list(app.last_rgb.shape)
                result["detector_label"] = app.confidence_label.cget("text")
                app.update_idletasks()
                ImageGrab.grab(window=app.winfo_id()).save(folder / "reported_photo_gui.png")
                app.check_current_frame()
                phase = "manual"
            elif phase == "manual" and not app.frame_check_busy and app._frame_pinned:
                result["manual_plate"] = app.plate_label.cget("text").replace(" ", "")
                app.show_page("history")
                app.query_plate.set(case["plate"])
                app.query_date.set("")
                app.refresh_records()
                result["history_rows"] = len(app.history_table.get_children())
                export = folder / "gui_export.csv"
                gui.filedialog.asksaveasfilename = lambda **kwargs: str(export)
                app.export_records()
                with export.open(encoding="utf-8-sig", newline="") as file:
                    exported = list(csv.DictReader(file))
                result["filtered_export"] = bool(exported) and all(r["plate"] == case["plate"] for r in exported)
                app.show_page("rules")
                app.rule_plate.set("1113")
                app.rule_date.set("2026-11-18")
                app.rule_time.set("12:00")
                app.check_rule()
                result["rule"] = app.last_rule_result.code
                app.video_stride_var.set("7")
                app.playback_speed_var.set("1.5")
                result["settings_saved"] = app.apply_settings()
                saved = load_settings(gui.DATA_DIR)
                result["settings_roundtrip"] = saved["frame_stride"] == 7 and saved["playback_speed"] == 1.5
                result["ok"] = (result["plate"] == case["plate"] == result["manual_plate"]
                                and result["history_rows"] > 0 and result["filtered_export"]
                                and result["rule"] == "restricted" and result["settings_roundtrip"])
                app.close()
                return
        except Exception as exc:
            result["error"] = str(exc)
            app.close()
            return
        app.after(80, inspect)

    app.after(200, lambda: app.start_source("image", [case["source"]]))
    app.after(300, inspect)
    app.mainloop()
    return result


def main():
    import cv2
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("test_artifacts/audit_2_8"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--mode", choices=("fast", "balanced", "detailed"), default="balanced")
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(folder / "verification.log"), level=logging.INFO, encoding="utf-8")
    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    settings = load_settings()
    settings.update(recognition_mode=args.mode, frame_stride=10)
    settings["rules"] = deepcopy(DEFAULTS["rules"])
    report = {"version": __version__, "mode": args.mode, "ok": False, "images": [], "negative_images": []}
    recognizer = PlateRecognizer(settings)
    store = RecordStore(folder / "records")
    for case in cases:
        started = time.monotonic()
        worker = DetectionWorker(settings, store)
        worker.recognizer = recognizer
        worker.start("image", [case["source"]])
        try:
            records = worker_result(worker)
            plates = [r["plate"] for r in records]
            expected = case.get("plates", [case["plate"]] if case.get("plate") else [])
            evidence_ok = True
            for record in records:
                evidence_ok &= all(store.resolve(record[key]).is_file() for key in ("image_path", "crop_path", "metadata_path"))
                metadata = json.loads(store.resolve(record["metadata_path"]).read_text(encoding="utf-8"))
                evidence_ok &= metadata["plate"] == record["plate"] and len(metadata["bbox"]) == 4
            row = {**case, "expected": expected, "plates": plates, "ok": set(expected) == set(plates) and evidence_ok,
                   "seconds": round(time.monotonic() - started, 2)}
        except Exception as exc:
            row = {**case, "ok": False, "error": str(exc)}
        report["images"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    for name in ("blank", "panel", "ordinary_text"):
        frame = np.full((400, 600, 3), 35, np.uint8)
        if name != "blank":
            cv2.rectangle(frame, (80, 80), (520, 330), (230, 230, 230), -1)
        if name == "ordinary_text":
            cv2.putText(frame, "WELCOME", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 10, 10), 3)
            cv2.putText(frame, "OFFICE", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 10, 10), 3)
        detections = recognizer.detect_image(frame)
        row = {"name": name, "plates": [d.plate for d in detections if d.plate]}
        row["ok"] = not row["plates"]
        report["negative_images"].append(row)

    report["gui"] = verify_gui(settings, recognizer, store, cases[0], folder)
    store.export(folder / "all_records.csv", store.records())
    if args.video:
        worker = DetectionWorker(settings, store)
        worker.settings["_video_review"] = True
        worker.recognizer = recognizer
        started = time.monotonic()
        worker.start("video", args.video)
        try:
            records = worker_result(worker, timeout=180)
            plates = {r["plate"] for r in records}
            expected = {"7034УБУ", "6657УНА", "4000УЕО"}
            report["video"] = {"plates": sorted(plates), "frames": worker.processed,
                               "seconds": round(time.monotonic() - started, 2),
                               "ok": expected == plates}
        except Exception as exc:
            report["video"] = {"ok": False, "error": str(exc)}
    if args.camera:
        worker = DetectionWorker(settings, RecordStore(folder / "camera_records"))
        worker.recognizer = recognizer
        started = time.monotonic()
        result = {"ok": False, "preview_frames": 0, "inference_frames": 0, "errors": []}
        worker.start("camera", "auto")
        try:
            while time.monotonic() - started < 40:
                try:
                    frame = worker.frames.get(timeout=.1)
                    result["preview_frames"] += 1
                    result.setdefault("shape", list(frame.shape))
                except queue.Empty:
                    pass
                while not worker.events.empty():
                    kind, value = worker.events.get_nowait()
                    if kind == "metrics":
                        result["inference_frames"] += 1
                    elif kind in ("error", "analysis_error"):
                        result["errors"].append(value)
                if result["inference_frames"] >= 3 or result["errors"]:
                    break
        finally:
            worker.stop()
            worker.thread.join(10)
        result["released"] = not worker.alive
        if result["released"]:
            try:
                capture, frame, _, _ = open_camera("auto")
                result["reopened"] = bool(frame.size)
                capture.release()
            except Exception as exc:
                result["errors"].append(str(exc))
        result["ok"] = bool(result["preview_frames"] and result["inference_frames"] >= 3
                            and result.get("reopened") and not result["errors"])
        report["camera"] = result
    checks = report["images"] + report["negative_images"] + [report["gui"]]
    checks.extend(report[key] for key in ("video", "camera") if key in report)
    report["ok"] = all(check["ok"] for check in checks)
    atomic_json(folder / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
