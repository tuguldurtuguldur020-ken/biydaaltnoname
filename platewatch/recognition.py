"""YOLO license-plate localisation and Mongolian EasyOCR recognition."""
from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import urllib.request
from .ocr import create_reader, text_regions
from .cyrillic_ocr import CyrillicReader
from .geometry import plate_quadrilaterals
from .plates import ALLOWLIST, Detection, display_plate, normalize_plate, select_ocr_candidate
from .rules import evaluate
from .settings import APP_DIR, DATA_DIR, now_ub

MODEL_REVISION = "83c98fbe7412fe8b3950adb5637cfd08b0f04809"
MODEL_URL = f"https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/{MODEL_REVISION}/best.pt"
MODEL_SHA256 = "2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822"


def plate_text_band(crop):
    """Remove dark bumper/grille padding around a light plate, when identifiable."""
    import cv2
    import numpy as np
    h, w = crop.shape[:2]
    if h < 8 or w < 24:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    left = np.percentile(gray[:, max(0, round(w * .05)):max(1, round(w * .16))], 85, axis=1)
    right = np.percentile(gray[:, min(w - 1, round(w * .84)):max(1, round(w * .95))], 85, axis=1)
    rows = (np.minimum(left, right) > threshold * 1.05).astype(np.uint8)[:, None]
    rows = cv2.morphologyEx(rows, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).ravel()
    changes = np.diff(np.r_[0, rows, 0].astype(int))
    runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    if not runs:
        return crop
    top, bottom = max(runs, key=lambda pair: pair[1] - pair[0])
    if not max(4, .3 * h) <= bottom - top < .9 * h:
        return crop
    return crop[max(0, top - 1):min(h, bottom + 1), max(0, round(w * .03)):max(1, round(w * .97))]


def crop_variants(crop, detailed=False):
    """Bounded, non-generative OCR preprocessing; the source pixels stay intact."""
    import cv2
    height, width = crop.shape[:2]
    if not height or not width:
        return
    scale = min(8.0 if detailed else 4.0, max(1.0, (144 if detailed else 100) / height),
                1800 / width)
    resized = cv2.resize(crop, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    yield gray
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield contrast
    if detailed:
        blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
        yield cv2.addWeighted(contrast, 1.5, blur, -.5, 0)
        yield cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def merge_detections(detections):
    """Merge overlapping full-frame/tile boxes without merging separate cars."""
    kept = []
    for item in sorted(detections, key=lambda d: (bool(d.plate), getattr(d, "reading_support", 1), d.ocr_confidence, d.detector_confidence), reverse=True):
        x1, y1, x2, y2 = item.bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        duplicate = False
        for other in kept:
            a, b, c, d = other.bbox
            intersection = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            if intersection / min(area, max(1, (c - a) * (d - b))) > .55:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def ensure_detector(settings, report=lambda text: None, cancel=None):
    if settings.get("model_path"):
        path = Path(settings["model_path"])
        if not path.is_file():
            raise RuntimeError("Сонгосон YOLO загвар олдсонгүй. Тохиргооноос .pt загвараа дахин сонгоно уу.")
        return path
    bundled = APP_DIR / "models" / "plate_detector.pt"
    target = bundled if bundled.exists() else DATA_DIR / "models" / "plate_detector.pt"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".download")
    report("Дугаар илрүүлэх загвар татаж байна (6.25 MB)...")
    try:
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "PlateWatch/2.0"})
        with urllib.request.urlopen(request, timeout=30) as response, part.open("wb") as file:
            while True:
                if cancel and cancel.is_set():
                    raise InterruptedError("Таталтыг зогсоолоо.")
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                file.write(chunk)
        if hashlib.sha256(part.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Загварын SHA-256 тохирсонгүй. Дахин татна уу.")
        part.replace(target)
    except Exception as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, InterruptedError):
            raise
        raise RuntimeError("YOLO загвар татаж чадсангүй. Интернэтээ шалгах эсвэл Тохиргооноос өөрийн .pt загварыг сонгоно уу.\n" + str(exc)) from exc
    return target


@lru_cache(maxsize=8)
def overlay_font(size):
    from PIL import ImageFont
    for name in [str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "segoeuib.ttf"), "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def annotate(frame, detections, rules, at=None):
    """Pillow renders Cyrillic labels correctly (cv2.putText cannot). Returns RGB."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_size = max(16, min(28, image.width // 45))
    font = overlay_font(font_size)
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        rule = evaluate(detection.plate, rules, at)
        color = "#fb7185" if rule.restricted_now else "#33dbc4" if detection.plate else "#fbbf24"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 500))
        label = f"{display_plate(detection.plate)}  {detection.ocr_confidence:.0%}" if detection.plate else "Дугаар • баталгаажаагүй"
        if rule.restricted_now:
            label += " • ХЯЗГААРЛАЛТТАЙ"
        box = draw.textbbox((0, 0), label, font=font)
        width, height = box[2] + 14, font_size + 12
        top = max(0, y1 - height)
        left = min(x1, max(0, image.width - width))
        draw.rounded_rectangle((left, top, left + width, top + height), radius=4, fill=color)
        draw.text((left + 7, top + 3), label, font=font, fill="#09131f")
    return np.asarray(image)


class PlateRecognizer:
    def __init__(self, settings, report=lambda text: None, cancel=None):
        model_path = ensure_detector(settings, report, cancel)
        report("YOLO загвар ачаалж байна...")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(DATA_DIR / "ultralytics"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        try:
            import torch
            from ultralytics import YOLO
            import easyocr
        except (ImportError, OSError) as exc:
            raise RuntimeError("AI сан ачаалж чадсангүй. run.bat-аа дахин ажиллуулна уу.\nWindows DLL хориглосон бол Windows Security / Application Control тохиргоог админтай шалгана уу.\n" + str(exc)) from exc
        self.settings = settings
        self.cancel = cancel
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 2)))
        self.device = "0" if settings.get("device") == "auto" and torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path), task="detect")
        names = list(self.model.names.values())
        self.class_ids = [i for i, name in self.model.names.items() if "plate" in name.lower() or "licen" in name.lower() or "дугаар" in name.lower()]
        if not self.class_ids and len(names) == 1:
            self.class_ids = [0]
        if not self.class_ids:
            raise RuntimeError("Энэ YOLO загвар дугаарын class агуулаагүй. License plate илрүүлэхээр сургасан .pt загвар сонгоно уу.")
        report("Монгол OCR ачаалж байна. Анх удаа жин татах тул интернэт хэрэгтэй...")
        cache = DATA_DIR / "models" / "ocr"
        cache.mkdir(parents=True, exist_ok=True)
        self.reader = create_reader(cache, gpu=self.device != "cpu")
        report("Кирилл үсгийн нэмэлт OCR ачаалж байна...")
        self.cyrillic_reader = CyrillicReader(cache / "cyrillic_ppocr", report, cancel)
        report("YOLO + Кирилл/Монгол OCR бэлэн")

    def read_crop(self, crop, detailed=False):
        self._last_read_support = 1
        primary = ("", 0.0, "")
        if hasattr(self, "cyrillic_reader"):
            self.cyrillic_reader.cancel = getattr(self, "cancel", None)
            primary = self.cyrillic_reader.read_plate(crop, detailed=detailed,
                                                     fast=self.settings.get("recognition_mode") == "fast" and not detailed)
            if primary[0] and primary[1] >= .85:
                return primary
            if hasattr(self, "reader") and (detailed or crop.shape[1] >= 100):
                agreement = self.read_agreeing_groups(crop)
                if agreement[0]:
                    self._last_read_support = 2
                    return agreement
            if primary[0] and primary[1] >= .65:
                return primary
        # Read the complete crop first. A cropped text band can clip a digit
        # and still produce a confidently wrong, well-formed plate.
        legacy_detailed = detailed and not hasattr(self, "cyrillic_reader")
        full = self._read_crop_text(crop, legacy_detailed)
        if detailed:
            refined = plate_text_band(crop)
            if not full[0] and refined.shape != crop.shape:
                best = self._read_crop_text(refined, detailed=legacy_detailed)
                full = max((best, full), key=lambda value: (bool(value[0]), value[1]))
        primary_plate = primary[0] or normalize_plate(primary[2])
        if primary_plate and primary[1] >= .5 and full[0] and full[0] != primary_plate:
            # Two trained recognizers disagree on a letter: do not present the
            # more confident one as established truth on a single source frame.
            return "", 0.0, primary[2] + " / " + full[2]
        if not full[0] and primary[0] and detailed and primary[1] >= max(.5, float(self.settings["ocr_confidence"])):
            # Detailed PP-OCR readings already require matching whole strings
            # from two views. An empty fallback must not erase that evidence.
            return primary
        return full

    def read_agreeing_groups(self, crop):
        """Compare both models on complete text groups separated by an emblem."""
        for variant in crop_variants(crop):
            if getattr(self, "cancel", None) is not None and self.cancel.is_set():
                raise InterruptedError()
            regions = text_regions(variant)
            if not 2 <= len(regions) <= 4:
                continue
            first = self.reader.recognize(variant, horizontal_list=regions, free_list=[], detail=1,
                                          paragraph=False, allowlist=ALLOWLIST, workers=0, batch_size=1)
            legacy = select_ocr_candidate(first)
            if not legacy[0]:
                continue
            second = []
            for x1, x2, y1, y2 in regions:
                raw, score = self.cyrillic_reader.read_line(variant[y1:y2, x1:x2])
                second.append(([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], raw, score))
            primary = select_ocr_candidate(second)
            joint_score = (primary[1] + legacy[1]) / 2
            if (primary[0] == legacy[0] and min(primary[1], legacy[1]) >= .2
                    and joint_score >= float(self.settings["ocr_confidence"])):
                return legacy[0], joint_score, legacy[2]
        return "", 0.0, ""

    def _read_crop_text(self, crop, detailed=False):
        best = ("", 0.0, "")
        votes = {}
        for index, variant in enumerate(crop_variants(crop, detailed)):
            if index and not detailed and getattr(self, "settings", {}).get("recognition_mode") == "fast":
                break
            if getattr(self, "cancel", None) is not None and self.cancel.is_set():
                raise InterruptedError()
            regions = text_regions(variant)
            full_plate = [[0, variant.shape[1], 0, variant.shape[0]]]
            # Prefer all original plate pixels. Contour groups are a fallback
            # for emblems or country labels which interrupt the text line.
            for boxes in ([full_plate, regions] if regions else [full_plate]):
                items = self.reader.recognize(variant, horizontal_list=boxes, free_list=[], detail=1,
                                               paragraph=False, allowlist=ALLOWLIST, workers=0, batch_size=1)
                candidate = select_ocr_candidate(items)
                if candidate[1] > best[1] or (not best[0] and candidate[2]):
                    best = candidate
                if candidate[0]:
                    votes.setdefault(candidate[0], {})[index] = candidate
                if candidate[0] and candidate[1] >= .8:
                    break
            if detailed and index < 2 and not best[0]:
                split = self.read_split_crop(variant)
                if split[0] and split[1] > best[1]:
                    best = split
                    votes.setdefault(split[0], {})[index] = split
            if detailed:
                supported = [entries for entries in votes.values() if len(entries) >= 2]
                if supported:
                    winner = max(supported, key=lambda entries: (len(entries), sum(v[1] for v in entries.values())))
                    weight = sum(value[1] for value in winner.values())
                    total = sum(value[1] for entries in votes.values() for value in entries.values())
                    if weight >= total * .7:
                        # Preserve the weaker score: agreement is not a probability.
                        return min(winner.values(), key=lambda value: value[1])
            elif best[0] and best[1] >= 0.8:
                break
        if detailed:
            return "", best[1], best[2]
        return best

    def read_split_crop(self, gray):
        """Read four digits and three letters separately around a country emblem."""
        best = ("", 0.0, "")
        height, width = gray.shape
        for number_end, letter_start in ((.55, .63), (.60, .65), (.50, .59)):
            parts = (gray[:, :max(1, round(width * number_end))], gray[:, round(width * letter_start):])
            texts, scores = [], []
            for part, allowed in zip(parts, ("0123456789", ALLOWLIST[10:])):
                if getattr(self, "cancel", None) is not None and self.cancel.is_set():
                    raise InterruptedError()
                items = self.reader.recognize(part, horizontal_list=[[0, part.shape[1], 0, height]],
                                              free_list=[], detail=1, paragraph=False, allowlist=allowed,
                                              workers=0, batch_size=1)
                items = sorted(items, key=lambda item: min(p[0] for p in item[0]))
                texts.append("".join(str(item[1]) for item in items).replace(" ", ""))
                scores.extend(float(item[2]) for item in items)
            if len(texts[0]) != 4 or len(texts[1]) != 3 or not scores:
                continue
            raw = "".join(texts)
            plate = normalize_plate(raw)
            confidence = min(scores)
            if plate and confidence > best[1]:
                best = plate, confidence, raw
            if best[0] and best[1] >= .8:
                break
        return best

    def detect(self, frame):
        found = self._detect_region(frame, 640)
        if not any(d.plate for d in found):
            found.extend(self._detect_geometry(frame))
        return merge_detections(found)

    def _detect_geometry(self, frame, video=False):
        found = []
        for bbox, crop in plate_quadrilaterals(frame, limit=1 if video else 3, minimum_width=80 if video else 40):
            if getattr(self, "cancel", None) is not None and self.cancel.is_set():
                raise InterruptedError()
            plate, confidence, raw = self.read_crop(crop, detailed=True)
            if not plate or confidence < max(.75, float(self.settings["ocr_confidence"])):
                continue
            # Geometric proposals have no learned detector score. Require a
            # second OCR preprocessing agreement before accepting the proposal.
            checked = self._read_crop_text(crop, detailed=True)
            if checked[0] != plate or checked[1] < .65:
                continue
            found.append(Detection(bbox, plate, 0.0, min(confidence, checked[1]), raw, "geometry_ocr",
                                   getattr(self, "_last_read_support", 1)))
        return found

    def detect_image(self, frame):
        return self.detect_detailed(frame)

    def detect_detailed(self, frame):
        found = self._detect_region(frame, 640, detailed=True)
        if not found and max(frame.shape[:2]) >= 800:
            found.extend(self._detect_region(frame, 960, detailed=True))
        if not any(d.plate for d in found):
            found.extend(self._detect_geometry(frame))
        h, w = frame.shape[:2]
        for item in list(found)[:4]:
            if item.plate and (item.ocr_confidence >= .7 or item.reading_support >= 2):
                continue
            x1, y1, x2, y2 = item.bbox
            pad_x, pad_y = x2 - x1, (y2 - y1) * 2
            left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
            right, bottom = min(w, x2 + pad_x), min(h, y2 + pad_y)
            refined = self._detect_region(frame[top:bottom, left:right], 640, detailed=True)
            for candidate in refined:
                a, b, c, d = candidate.bbox
                candidate.bbox = (a + left, b + top, c + left, d + top)
            found.extend(refined)
        # Overlapping native-pixel tiles make small plates larger at the detector
        # input. No downloaded video or saved source frame is resampled in place.
        if not found:
            found.extend(self._detect_tiles(frame))
        return merge_detections(found)

    def _detect_tiles(self, frame, video=False):
        h, w = frame.shape[:2]
        found = []
        if min(h, w) >= 200 and max(h, w) >= 400:
            tile_w, tile_h = round(w * .6), round(h * .6)
            for top in (0, h - tile_h):
                for left in (0, w - tile_w):
                    cancel = getattr(self, "cancel", None)
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError()
                    items = self._detect_region(frame[top:top + tile_h, left:left + tile_w], 640, detailed=True, video=video)
                    for item in items:
                        x1, y1, x2, y2 = item.bbox
                        item.bbox = (x1 + left, y1 + top, x2 + left, y2 + top)
                    found.extend(items)
        return found

    def detect_video_frame(self, frame):
        # Live/video readings are confirmed across source frames by the worker.
        # Expensive preprocessing is an explicit user choice.
        detailed = self.settings.get("recognition_mode", "balanced") == "detailed"
        found = self._detect_region(frame, 960 if detailed else 640, detailed=detailed, video=True)
        if detailed and not any(d.plate for d in found):
            found.extend(self._detect_tiles(frame, video=True))
        if not any(d.plate for d in found):
            found.extend(self._detect_geometry(frame, video=True))
        return merge_detections(found)

    def _detect_region(self, frame, image_size, detailed=False, video=False):
        result = None
        for size in dict.fromkeys((image_size, 320)):
            try:
                result = self.model.predict(frame, conf=float(self.settings["detection_confidence"]),
                                            classes=self.class_ids, device=self.device, imgsz=size, max_det=15, verbose=False)[0]
                break
            except (MemoryError, RuntimeError) as exc:
                if size == 320 or (not isinstance(exc, MemoryError) and "memory" not in str(exc).lower()):
                    raise
                logging.warning("Detector allocation failed; retrying at 320 pixels")
            import gc
            gc.collect()
        found = []
        if result.boxes is None:
            return found
        h, w = frame.shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pad_x, pad_y = max(2, int((x2 - x1) * .04)), max(2, int((y2 - y1) * .08))
            x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            # A two-row plate needs fewer horizontal pixels than a seven-symbol
            # single row. The old 45 px cutoff discarded readable square plates.
            if video and (x2 - x1 < 28 or y2 - y1 < 9):
                found.append(Detection((x1, y1, x2, y2), detector_confidence=float(box.conf[0])))
                continue
            read = (lambda crop: self.read_crop(crop, detailed=True)) if detailed else self.read_crop
            plate, confidence, raw = read(frame[y1:y2, x1:x2])
            support = getattr(self, "_last_read_support", 1)
            # A generic detector sometimes boxes only the four digits on a
            # Mongolian plate. Retry a wider region to include the three letters.
            if not plate and x2 - x1 < 4.5 * (y2 - y1):
                target_width = min(w, int((y2 - y1) * 5.4))
                left = max(0, min(w - target_width, int((x1 + x2 - target_width) / 2)))
                right = left + target_width
                if right - left > x2 - x1:
                    wider = read(frame[y1:y2, left:right])
                    if wider[0]:
                        plate, confidence, raw = wider
                        support = getattr(self, "_last_read_support", 1)
                        x1, x2 = left, right
            if confidence < float(self.settings["ocr_confidence"]):
                plate = ""
            found.append(Detection((x1, y1, x2, y2), plate, float(box.conf[0]), confidence, raw,
                                   reading_support=support))
        return found
