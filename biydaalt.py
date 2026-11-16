"""PlateWatch desktop entry point. Double-click run.bat on Windows."""
import multiprocessing
import argparse
import sys


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="PlateWatch desktop plate recognition")
    parser.add_argument("--smoke-test", action="store_true")
    source_args = parser.add_mutually_exclusive_group()
    source_args.add_argument("--image", help="Open an image when the GUI starts")
    source_args.add_argument("--video", help="Play and check a video when the GUI starts")
    source_args.add_argument("--camera", nargs="?", const="auto", help="Start camera preview; auto-detect by default")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Rotate the selected source clockwise before playback and recognition")
    parser.add_argument("--verify-image", help="Check packaged YOLO/OCR and write a JSON report")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    if args.verify_image:
        from platewatch.diagnostics import verify_image
        return verify_image(args.verify_image, args.report)
    from contextlib import nullcontext
    from platewatch.instance import desktop_instance
    from platewatch.gui import PlateWatchApp
    with nullcontext(True) if args.smoke_test else desktop_instance() as first:
        if not first:
            print("PlateWatch is already open. Close the existing window and run run.bat again to load an update.", flush=True)
            if args.camera is not None or args.image or args.video:
                from platewatch.settings import DATA_DIR, atomic_json, now_ub
                request = ({"kind": "camera", "source": args.camera} if args.camera is not None else
                           {"kind": "video", "source": args.video} if args.video else
                           {"kind": "image", "source": [args.image]})
                request["at"] = now_ub().timestamp()
                request["rotation"] = args.rotate
                atomic_json(DATA_DIR / "open_request.json", request)
            return 0
        app = PlateWatchApp()
        app.deiconify()
        app.lift()
        if args.image:
            app.after(200, lambda: app.start_source("image", [args.image], args.rotate))
        elif args.video:
            app.after(200, lambda: app.start_source("video", args.video, args.rotate))
        elif args.camera is not None:
            app.camera_var.set("Авто" if args.camera.lower() == "auto" else args.camera)
            app.after(200, lambda: app.start_source("camera", args.camera, args.rotate))
        if args.smoke_test:
            app.after(1600, app.close)
        app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
