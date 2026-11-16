"""Position-aware OCR cleanup for common Mongolian 4 digit + 3 letter plates."""
from dataclasses import dataclass
from collections import deque
import re

CYRILLIC = "АБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ"
ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" + CYRILLIC
LATIN_TO_CYR = str.maketrans({"A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У"})
DIGIT_CONFUSIONS = str.maketrans({"O": "0", "О": "0", "I": "1", "І": "1", "З": "3"})
OCR_LOOKALIKES = str.maketrans({"Δ": "Д", "∆": "Д", "Α": "А", "Β": "В", "Ε": "Е", "Η": "Н",
                              "Κ": "К", "Μ": "М", "Ο": "О", "Ρ": "Р", "Τ": "Т", "Υ": "У", "Χ": "Х"})


def clean_text(text):
    return re.sub(r"[^0-9A-Z" + CYRILLIC + r"]", "", str(text).upper().translate(OCR_LOOKALIKES))


def normalize_plate(text):
    value = clean_text(text)
    if len(value) != 7:
        return ""
    digits = value[:4].translate(DIGIT_CONFUSIONS)
    letters = value[4:].translate(LATIN_TO_CYR)
    if re.fullmatch(r"[0-9]{4}", digits) and re.fullmatch("[" + CYRILLIC + "]{3}", letters):
        return digits + letters
    return ""


def display_plate(text):
    return text[:4] + " " + text[4:] if len(text) == 7 else text


def select_ocr_candidate(items):
    """Join separate OCR boxes in reading order; preserve conservative confidence."""
    if not items:
        return "", 0.0, ""
    ordered = sorted(items, key=lambda item: (sum(p[1] for p in item[0]) / 4, min(p[0] for p in item[0])))
    heights = [max(p[1] for p in i[0]) - min(p[1] for p in i[0]) for i in items]
    centers = [sum(p[1] for p in i[0]) / 4 for i in items]
    if max(centers) - min(centers) <= max(heights) * 0.65:
        ordered = sorted(items, key=lambda item: min(p[0] for p in item[0]))
    candidates = []
    for start in range(len(ordered)):
        for end in range(start + 1, min(len(ordered), start + 4) + 1):
            group = ordered[start:end]
            raw = "".join(str(item[1]) for item in group)
            plate = normalize_plate(raw)
            if plate:
                candidates.append((plate, min(float(item[2]) for item in group), raw))
    return max(candidates, key=lambda value: value[1]) if candidates else ("", 0.0, " ".join(str(i[1]) for i in ordered))


@dataclass
class Detection:
    bbox: tuple
    plate: str = ""
    detector_confidence: float = 0.0
    ocr_confidence: float = 0.0
    raw_text: str = ""
    method: str = "yolo_ocr"
    reading_support: int = 1


class PlateTracker:
    """Confirm a whole plate on the same moving object, using distinct frames.

    Keep conflicting readings as evidence, but never substitute a historical
    number onto a frame which currently reads something else.
    """
    def __init__(self):
        self.tracks = []

    def confirm(self, detections, clock, confirmations=2, confidence=.65, window=3):
        from dataclasses import replace
        self.tracks = [t for t in self.tracks if 0 <= clock - t["time"] <= window]
        available = list(self.tracks)
        output = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            matches = []
            for track in available:
                a, b, c, d = track["bbox"]
                tw, th = max(1, c - a), max(1, d - b)
                dx, dy = abs(x1 + x2 - a - c) / 2, abs(y1 + y2 - b - d) / 2
                intersection = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
                overlap = intersection / (w * h + tw * th - intersection)
                # Approaching cars can move over a plate-width between sparse
                # video samples; choose the closest compatible box in that lane.
                if .4 <= w / tw <= 2.5 and .4 <= h / th <= 2.5 and (overlap >= .15 or (dx < max(w, tw) * 2 and dy < max(h, th) * 3)):
                    matches.append((overlap - (dx / max(w, tw) + dy / max(h, th)) * .1, track))
            if matches:
                track = max(matches, key=lambda pair: pair[0])[1]
                available = [t for t in available if t is not track]
            else:
                track = {"votes": deque(maxlen=max(15, confirmations * 2)), "time": clock, "bbox": detection.bbox}
                self.tracks.append(track)
            # One vote per object per source frame, even after a seek/recheck.
            if clock != track.get("voted_at"):
                plate = detection.plate if detection.ocr_confidence >= confidence else ""
                track["votes"].append((plate, detection.ocr_confidence))
                track["voted_at"] = clock
            track.update(time=clock, bbox=detection.bbox)
            votes = list(track["votes"])
            scores, counts = {}, {}
            for plate, score in votes:
                if plate:
                    scores[plate] = scores.get(plate, 0) + score
                    counts[plate] = counts.get(plate, 0) + 1
            plate = detection.plate
            supported = (plate and counts.get(plate, 0) >= confirmations
                         and scores.get(plate, 0) >= sum(scores.values()) * .7
                         and detection.ocr_confidence >= confidence)
            track["pending"] = (not supported and bool(plate or re.search(r"\d{4}", detection.raw_text)))
            output.append(detection if supported else replace(detection, plate=""))
        return output


class TemporalGate:
    """Confirm repeated readings in a short window; suppress duplicate records."""
    def __init__(self, confirmations=2, cooldown=20, window=8):
        self.confirmations = confirmations
        self.cooldown = cooldown
        self.window = window
        self.hits = {}
        self.saved = {}

    def ready(self, plate, clock, single_image=False):
        if not plate:
            return False
        if single_image:
            return True
        if clock - self.saved.get(plate, float("-inf")) < self.cooldown:
            return False
        hits = [t for t in self.hits.get(plate, []) if clock - t <= self.window]
        hits.append(clock)
        self.hits[plate] = hits[-self.confirmations:]
        return len(hits) >= self.confirmations

    def mark_saved(self, plate, clock):
        self.saved[plate] = clock
        self.hits.pop(plate, None)
        if len(self.saved) > 5000:
            self.saved = {p: t for p, t in self.saved.items() if clock - t < self.cooldown}
        if len(self.hits) > 1000:
            self.hits = {p: times for p, times in self.hits.items() if times and clock - times[-1] <= self.window}
