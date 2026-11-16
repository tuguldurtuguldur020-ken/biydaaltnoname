"""Select a fixture through the actual Windows file dialog and run real AI."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import threading
import time
from PIL import ImageGrab
from platewatch.gui import PlateWatchApp
from platewatch.settings import atomic_json
from platewatch.storage import RecordStore
from platewatch.worker import DetectionWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("test_artifacts/synthetic_car.png"))
    parser.add_argument("--camera-first", action="store_true")
    parser.add_argument("--simulate-render-failure", action="store_true")
    args = parser.parse_args()
    image = args.image.resolve()
    if not image.is_file():
        parser.error("fixture image not found")
    app = PlateWatchApp()
    app.store = RecordStore(Path("test_artifacts/native_import_evidence"))
    app.worker = DetectionWorker(app.settings, app.store)
    before = len(app.store.records())
    report = {"native_picker_found": False, "filename_set": False, "ok": False}
    if args.simulate_render_failure:
        render = app._render_preview
        def fail_once():
            if app.last_rgb is not None and not report.get("injected_preview_memory_error"):
                report["injected_preview_memory_error"] = True
                raise MemoryError("Verification: recover from the original camera freeze")
            render()
        app._render_preview = fail_once
    user = ctypes.WinDLL("user32", use_last_error=True)
    callback = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    user.SendMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
    user.PostMessageW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]

    def windows(parent=None):
        result = []
        @callback
        def visit(handle, param):
            pid = w.DWORD()
            user.GetWindowThreadProcessId(handle, ctypes.byref(pid))
            if pid.value == os.getpid():
                result.append(handle)
            return True
        user.EnumWindows(visit, 0) if parent is None else user.EnumChildWindows(parent, visit, 0)
        return result

    def class_name(handle):
        name = ctypes.create_unicode_buffer(128)
        user.GetClassNameW(handle, name, 128)
        return name.value

    def choose_file():
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            dialogs = [h for h in windows() if class_name(h) == "#32770" and user.IsWindowVisible(h)]
            if not dialogs:
                time.sleep(.1)
                continue
            dialog = dialogs[0]
            report["native_picker_found"] = True
            edits = [h for h in windows(dialog) if class_name(h) == "Edit" and user.IsWindowVisible(h)]
            report["edit_ids"] = [user.GetDlgCtrlID(h) for h in edits]
            if not edits:
                time.sleep(.15)
                continue
            edit = next((h for h in edits if user.GetDlgCtrlID(h) in (1148, 1001)), edits[-1])
            text = ctypes.create_unicode_buffer(str(image))
            user.SendMessageW(edit, 0x000C, 0, ctypes.addressof(text))
            report["filename_set"] = True
            user.PostMessageW(dialog, 0x0111, 1, 0)  # IDOK in the Windows Open dialog
            return
        report["error"] = "Windows file dialog did not expose its filename field"
        for handle in windows():
            if class_name(handle) == "#32770":
                user.PostMessageW(handle, 0x0111, 2, 0)

    def import_image():
        threading.Thread(target=choose_file, daemon=True).start()
        app.after(100, app.main_image_button.invoke)
    phase = {"value": "camera" if args.camera_first else "image"}
    if args.camera_first:
        app.after(250, app.camera_button.invoke)
    else:
        import_image()
    started = time.monotonic()

    def inspect():
        if phase["value"] == "camera" and app.preview_displayed:
            report.setdefault("real_camera_preview_seconds", round(time.monotonic() - started, 3))
            report["real_camera_shape"] = list(app.last_rgb.shape)
        if phase["value"] == "camera" and app.worker.processed >= 2 and app.preview_displayed:
            report["real_camera_inference_frames"] = app.worker.processed
            report["real_camera_shape"] = list(app.last_rgb.shape)
            report["preview_recovered"] = not app._preview_failed
            phase["value"] = "image"
            import_image()
        image_records = [r for r in app.store.records()[before:] if r["source_kind"] == "image"]
        if image_records and app.preview_displayed and app.active_kind == "image" and not app._preview_dirty:
            report.update(ok=True, plate=image_records[-1]["plate"], preview_shape=list(app.last_rgb.shape))
            report["app_version"] = app.title()
            report["preview_tk_dimensions"] = [app.preview_photo.width(), app.preview_photo.height()]
            if args.simulate_render_failure:
                report["ok"] = bool(report.get("injected_preview_memory_error")) and not app._preview_failed
            app.update_idletasks()
            ImageGrab.grab(window=app.winfo_id()).save("test_artifacts/native_import_gui.png")
            app.show_page("rules")
            app.rule_plate.set("1113")
            app.rule_date.set("2026-11-13")
            app.rule_time.set("12:00")
            app.check_rule()
            report["1113_schedule"] = app.rule_result_label.cget("text")
            report["1113_selected_date_result"] = app.last_rule_result.code
            report["ok"] = report["ok"] and "Лхагва" in report["1113_schedule"] and "Баасан" not in report["1113_schedule"] and app.last_rule_result.code == "allowed"
            app.update_idletasks()
            ImageGrab.grab(window=app.winfo_id()).save("test_artifacts/rules_1113_gui.png")
            app.close()
        elif time.monotonic() - started > 80 or app.last_error or report.get("error"):
            report["error"] = app.last_error or report.get("error", app.status_text.get())
            app.close()
        else:
            app.after(150, inspect)
    app.after(150, inspect)
    app.mainloop()
    atomic_json(Path("test_artifacts/native_import_verification.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
