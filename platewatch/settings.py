from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from . import __version__

APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PLATEWATCH_DATA_DIR", str(APP_DIR / "data")))
APP_TITLE = f"PlateWatch {__version__} • Автомашины дугаар танигч"
UB_TIME = timezone(timedelta(hours=8), "Asia/Ulaanbaatar")
_json_lock = threading.RLock()
DEFAULTS = {
    "model_path": "", "camera_index": 0, "camera_backend": "auto", "device": "cpu",
    "detection_confidence": 0.35, "ocr_confidence": 0.35,
    "frame_stride": 10, "cooldown_seconds": 20, "confirmation_frames": 2,
    "recognition_mode": "balanced", "video_min_confidence": .65,
    "playback_speed": 1.0, "neighbor_frames": 2,
    "adaptive_sampling": True,
    "accent": "Цэнхэр", "save_video": False,
    "rules": {
        "mode": "weekly", "start": "08:00", "end": "20:00",
        "in_zone": True, "even_on_even_date": True,
        "weekdays": {"0": [1, 6], "1": [2, 7], "2": [3, 8], "3": [4, 9], "4": [5, 0], "5": [], "6": []},
        "holidays": [], "exempt_plates": [],
    },
}


def now_ub():
    return datetime.now(UB_TIME)


def atomic_json(path, data):
    with _json_lock:
        _write_json(path, data)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as file:
            temp = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        for attempt in range(5):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.01 * 2 ** attempt)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def validated_settings(incoming):
    """Recover valid preferences without letting a malformed value break startup."""
    from .plates import normalize_plate
    result = deepcopy(DEFAULTS)
    if not isinstance(incoming, dict):
        return result
    numbers = {
        "detection_confidence": (.05, .95, float), "ocr_confidence": (.05, .95, float),
        "frame_stride": (1, 120, int), "cooldown_seconds": (1, 3600, int),
        "confirmation_frames": (1, 10, int), "video_min_confidence": (.05, .99, float),
        "playback_speed": (.25, 4, float), "neighbor_frames": (0, 10, int),
    }
    for key, (low, high, cast) in numbers.items():
        try:
            value = incoming[key]
            numeric = float(value)
            if (not isinstance(value, bool) and math.isfinite(numeric) and low <= numeric <= high
                    and (cast is not int or numeric.is_integer())):
                result[key] = cast(numeric)
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
    enums = {"recognition_mode": ("fast", "balanced", "detailed"), "device": ("cpu", "auto"),
             "camera_backend": ("auto", "dshow", "msmf"), "accent": ("Цэнхэр", "Ногоон", "Нил ягаан", "Ягаан")}
    for key, choices in enums.items():
        if incoming.get(key) in choices:
            result[key] = incoming[key]
    for key in ("save_video", "adaptive_sampling"):
        if isinstance(incoming.get(key), bool):
            result[key] = incoming[key]
    if isinstance(incoming.get("model_path"), str):
        result["model_path"] = incoming["model_path"]
    camera = incoming.get("camera_index")
    if camera in ("auto", "Авто"):
        result["camera_index"] = "auto"
    elif isinstance(camera, int) and not isinstance(camera, bool) and 0 <= camera <= 99:
        result["camera_index"] = camera
    rules = incoming.get("rules")
    if not isinstance(rules, dict):
        return result
    target = result["rules"]
    if rules.get("mode") in ("weekly", "odd_even", "off"):
        target["mode"] = rules["mode"]
    for key in ("in_zone", "even_on_even_date"):
        if isinstance(rules.get(key), bool):
            target[key] = rules[key]
    for key in ("start", "end"):
        try:
            value = datetime.strptime(rules[key], "%H:%M")
            target[key] = value.strftime("%H:%M")
        except (KeyError, TypeError, ValueError):
            pass
    if target["start"] == target["end"]:
        target.update(start=DEFAULTS["rules"]["start"], end=DEFAULTS["rules"]["end"])
    weekdays = rules.get("weekdays")
    if isinstance(weekdays, dict):
        for key in target["weekdays"]:
            digits = weekdays.get(key)
            if isinstance(digits, list) and all(type(n) is int and 0 <= n <= 9 for n in digits):
                target["weekdays"][key] = sorted(set(digits))
    holidays = rules.get("holidays")
    if isinstance(holidays, list):
        target["holidays"] = []
        for day in holidays:
            try:
                normalized = datetime.strptime(day, "%Y-%m-%d").date().isoformat()
                if normalized not in target["holidays"]:
                    target["holidays"].append(normalized)
            except (TypeError, ValueError):
                pass
    if isinstance(rules.get("exempt_plates"), list):
        target["exempt_plates"] = list(dict.fromkeys(p for v in rules["exempt_plates"] if isinstance(v, str) and (p := normalize_plate(v))))
    return result


def load_settings(directory=DATA_DIR):
    result = deepcopy(DEFAULTS)
    path = Path(directory) / "settings.json"
    if path.exists():
        try:
            incoming = json.loads(path.read_text(encoding="utf-8-sig"))
            result = validated_settings(incoming)
        except (ValueError, OSError):
            pass  # Preserve damaged settings for recovery.
    return result


def save_settings(settings, directory=DATA_DIR):
    atomic_json(Path(directory) / "settings.json", settings)
