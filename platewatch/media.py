"""Open local images and camera streams without depending on AI startup."""
from pathlib import Path
import math
import sys

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".tif", ".tiff", ".gif")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm")


def rotate_frame(frame, degrees=0):
    import cv2
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(frame, rotations[degrees]) if degrees in rotations else frame


def read_image(path):
    """Pillow handles Unicode filenames, EXIF orientation, TIFF and GIF frames."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Зургийн файл олдсонгүй: {path}")
    try:
        with Image.open(path) as original:
            rgb = np.asarray(ImageOps.exif_transpose(original).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Зураг нээж чадсангүй: {path.name}. JPG, PNG, WEBP, BMP, TIFF эсвэл GIF файл сонгоно уу.\n{exc}") from exc


def open_video(path, cancel=None):
    """Validate the source and try actual video decoders, never image sequences."""
    import cv2
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Видео файл олдсонгүй: {path}")
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        raise RuntimeError("Энэ нь зургийн файл байна. «+ Зураг» товчоор оруулна уу.")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError("Видео файл сонгоно уу: MP4, AVI, MOV, MKV, WEBM, WMV эсвэл M4V.")
    backends = [("FFMPEG", cv2.CAP_FFMPEG), ("AUTO", cv2.CAP_ANY)]
    if sys.platform == "win32":
        backends.insert(1, ("MSMF", cv2.CAP_MSMF))
    for name, api in backends:
        if cancel and cancel.is_set():
            raise InterruptedError()
        capture = None
        try:
            capture = cv2.VideoCapture(str(path), api)
            ok, frame = capture.read() if capture.isOpened() else (False, None)
            if not ok or frame is None or not frame.size:
                continue
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and 1 <= fps <= 240 else 25.0
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            total = max(0, int(count)) if math.isfinite(count) else 0
            info = {"fps": fps, "total": total, "duration": total / fps,
                    "backend": name, "shape": list(frame.shape)}
            result, capture = capture, None
            return result, frame, info
        except cv2.error:
            continue
        finally:
            if capture is not None:
                capture.release()
    raise RuntimeError(f"«{path.name}» видеоны дүрсийг уншиж чадсангүй.\n"
                       "Файл дутуу татагдсан эсэхийг шалгана уу. Өөр тоглуулагчид нээгддэг бол MP4 (H.264) болгон хөрвүүлээд оруулна уу.")


def open_camera(index="auto", backend="auto", report=lambda text: None, cancel=None):
    """Find a working camera, including when a stale/manual index is unavailable."""
    import cv2
    choices = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("AUTO", cv2.CAP_ANY)] if sys.platform == "win32" else [("AUTO", cv2.CAP_ANY)]
    if backend != "auto":
        choices.sort(key=lambda item: item[0].lower() != backend.lower())
    requested = None if index is None or str(index).lower() in ("auto", "авто") else int(index)
    indices = list(dict.fromkeys(([requested] if requested is not None else []) + [0, 1, 2, 3]))
    for candidate in indices:
        for name, api in choices:
            if cancel and cancel.is_set():
                raise InterruptedError()
            report(f"Камер хайж байна • төхөөрөмж {candidate} • {name}")
            capture = None
            try:
                capture = cv2.VideoCapture(candidate, api)
                if not capture.isOpened():
                    continue
                # Some Windows drivers need several warm-up reads before the
                # first frame, even though isOpened() already returns True.
                for _ in range(10):
                    if cancel and cancel.is_set():
                        raise InterruptedError()
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        result, capture = capture, None
                        return result, frame, name, candidate
                    if cancel:
                        cancel.wait(.05)
            except cv2.error:
                continue
            finally:
                if capture is not None:
                    capture.release()
    raise RuntimeError("Ажиллах камер олдсонгүй.\nКамер ашиглаж буй бусад програмыг хааж, Windows Camera permission болон камерын холболтоо шалгана уу.")
