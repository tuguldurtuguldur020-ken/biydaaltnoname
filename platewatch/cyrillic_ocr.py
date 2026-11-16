"""Small, local PP-OCRv5 Cyrillic recognizer; no image/text leaves the PC.

Model and preprocessing: PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx,
Apache-2.0. The pinned inference.yml contains the matching CTC dictionary.
"""
import hashlib
from pathlib import Path
import urllib.request

from .plates import clean_text, normalize_plate
from .layout import two_row_crops

REVISION = "2cef88145434beb8afa9dd82d77d799eb1ad7b29"
MODEL_BASE = f"https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx/resolve/{REVISION}/"
FILES = {
    "inference.onnx": "5371ee1ddaa7983cc62d0818d99e982b6804638c85e4f960d59a574094e172e5",
    "inference.yml": "5c76cc91fa98410178a09f498db10050d0ec1634a660053d3005ab7be581f501",
}


def ensure_cyrillic_model(directory, report=lambda text: None, cancel=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, checksum in FILES.items():
        target = directory / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == checksum:
            continue
        report("Кирилл OCR загвар татаж байна (8 MB, зөвхөн анхны удаа)...")
        part = target.with_suffix(target.suffix + ".download")
        try:
            with urllib.request.urlopen(MODEL_BASE + name, timeout=30) as response, part.open("wb") as out:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError()
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            if hashlib.sha256(part.read_bytes()).hexdigest() != checksum:
                raise RuntimeError("Кирилл OCR загварын SHA-256 тохирсонгүй.")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
    return directory


def light_plate(crop):
    """Keep the physical light plate instead of a detector's dark bumper padding."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    if h < 8 or w < 20:
        return crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 3), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [(x, y, a, b) for x, y, a, b in boxes
             if a * b >= w * h * .15 and a >= w * .4 and b >= 6 and 1.1 <= a / b <= 7]
    if not boxes:
        return crop
    x, y, a, b = max(boxes, key=lambda box: box[2] * box[3])
    if a * b >= w * h * .92:
        return crop
    return crop[y:y + b, x:x + a]


def select_reading(readings, minimum_views=1):
    """Different views support a whole string; never assemble letters from guesses."""
    votes = {}
    for view, candidate in readings:
        plate, confidence, raw = candidate
        if plate and confidence >= .5:
            by_view = votes.setdefault(plate, {})
            if confidence > by_view.get(view, ("", 0, ""))[1]:
                by_view[view] = candidate
    if not votes:
        return "", 0.0, readings[-1][1][2] if readings else ""
    plate, support = max(votes.items(), key=lambda item: (len(item[1]), sum(v[1] for v in item[1].values())))
    total = sum(v[1] for support in votes.values() for v in support.values())
    weight = sum(v[1] for v in support.values())
    best = min(support.values(), key=lambda value: value[1])
    if len(support) < minimum_views or weight < total * .7:
        return "", best[1], best[2]
    return plate, best[1], best[2]


class CyrillicReader:
    def __init__(self, directory, report=lambda text: None, cancel=None):
        import onnxruntime as ort
        import yaml
        directory = ensure_cyrillic_model(directory, report, cancel)
        config = yaml.safe_load((directory / "inference.yml").read_text(encoding="utf-8"))
        self.characters = [""] + config["PostProcess"]["character_dict"] + [" "]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(directory / "inference.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.cancel = cancel

    def read_line(self, crop):
        import cv2
        import numpy as np
        if self.cancel is not None and self.cancel.is_set():
            raise InterruptedError()
        h, w = crop.shape[:2]
        if not h or not w:
            return "", 0.0
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        width = min(1280, max(8, int(np.ceil(48 * w / h))))
        image = cv2.resize(crop, (width, 48)).astype(np.float32) / 127.5 - 1
        tensor = np.zeros((1, 3, 48, max(320, width)), np.float32)
        tensor[0, :, :, :width] = image.transpose(2, 0, 1)
        probabilities = self.session.run(None, {self.input_name: tensor})[0][0]
        indices = probabilities.argmax(axis=1)
        selection = np.r_[True, indices[1:] != indices[:-1]] & (indices != 0)
        raw = "".join(self.characters[i] for i in indices[selection])
        # Punctuation/emblems do not inflate or depress the plate text score.
        scores = [float(probabilities[t, i]) for t, i in enumerate(indices)
                  if selection[t] and clean_text(self.characters[i])]
        if not scores:
            return raw, 0.0
        # A weak final letter must not hide behind six very confident symbols.
        return raw, (sum(scores) / len(scores) + min(scores)) / 2

    def read_plate(self, crop, detailed=False, fast=False):
        import cv2
        readings = []
        tight = light_plate(crop)
        views = [("source", crop)]
        if tight.shape != crop.shape:
            views.append(("plate", tight))
        gray = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY) if tight.ndim == 3 else tight
        if not fast:
            views.append(("contrast", cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)))
        for view, image in views:
            raw, confidence = self.read_line(image)
            readings.append((view, (normalize_plate(raw), confidence, raw)))
            # Square plates have two real text rows. Read them separately, in
            # order, rather than stretching both rows into a single text line.
            h, w = image.shape[:2]
            if w / h < 2.5:
                for split in (.60, .55, .65):
                    cut = round(h * split)
                    number, n_score = self.read_line(image[:cut])
                    letters, l_score = self.read_line(image[cut:])
                    if len(clean_text(number)) == 4 and len(clean_text(letters)) == 3:
                        raw = number + letters
                        readings.append((view, (normalize_plate(raw), min(n_score, l_score), raw)))
            best = select_reading(readings, minimum_views=2 if detailed else 1)
            if best[0] and best[1] >= .8 and not detailed:
                return best
        best = select_reading(readings, minimum_views=2 if detailed else 1)
        if best[0]:
            return best
        # Fixed horizontal splits cannot separate slanted rows inside an
        # oversized detector box. Locate real character rows and straighten
        # each row before comparing complete readings from distinct views.
        for number, letters in two_row_crops(crop):
            row_readings = []
            for view in ("gray", "contrast"):
                texts, scores = [], []
                for row in (number, letters):
                    gray = cv2.cvtColor(row, cv2.COLOR_BGR2GRAY) if row.ndim == 3 else row
                    if view == "contrast":
                        gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4)).apply(gray)
                    raw, score = self.read_line(gray)
                    texts.append(raw)
                    scores.append(score)
                if len(clean_text(texts[0])) == 4 and len(clean_text(texts[1])) == 3:
                    raw = "".join(texts)
                    row_readings.append((view, (normalize_plate(raw), min(scores), raw)))
            reading = select_reading(row_readings, minimum_views=2)
            if reading[0]:
                return reading
        return best
