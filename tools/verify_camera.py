"""Verify the actual camera/AI worker without retaining ordinary camera frames."""
import argparse
import json
from pathlib import Path
import queue
import time
from platewatch.media import open_camera
from platewatch.settings import load_settings, atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="auto")
    parser.add_argument("--report", type=Path, default=Path("test_artifacts/camera_verification.json"))
    args = parser.parse_args()
    worker = DetectionWorker(load_settings(), RecordStore(Path("test_artifacts/camera_evidence")))
    report = {"kind": "real_camera", "index": args.index, "preview_frames": 0, "inference_frames": 0, "errors": []}
    started = time.monotonic()
    worker.start("camera", args.index)
    try:
        while time.monotonic() - started < 50:
            try:
                frame = worker.frames.get(timeout=.15)
                report["preview_frames"] += 1
                report.setdefault("first_preview_seconds", round(time.monotonic() - started, 3))
                report["frame_shape"] = list(frame.shape)
            except queue.Empty:
                pass
            while not worker.events.empty():
                kind, value = worker.events.get_nowait()
                if kind == "source_ready":
                    report["source"] = value
                elif kind == "metrics":
                    report["inference_frames"] += 1
                elif kind == "error":
                    report["errors"].append(value)
            if report["inference_frames"] >= 3 or report["errors"] or not worker.alive:
                break
    finally:
        worker.stop_event.set()
        worker.pause_event.clear()
        worker.thread.join(12)
    report["stopped"] = not worker.alive
    if report["stopped"]:
        try:
            capture, frame, backend, actual_index = open_camera(args.index)
            report["camera_reopened_after_stop"] = bool(frame.size)
            capture.release()
        except Exception as exc:
            report["errors"].append(str(exc))
    report["ok"] = bool(report["preview_frames"] and report["inference_frames"] and report["stopped"] and report.get("camera_reopened_after_stop") and not report["errors"])
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
