"""CSV index + JSON metadata + JPEG images; no SQL or database service."""
import csv
import logging
import math
from pathlib import Path
import threading
from uuid import uuid4
from .settings import DATA_DIR, atomic_json, now_ub
from .plates import normalize_plate

FIELDS = ["id", "detected_at", "plate", "source", "source_kind", "video_seconds", "detector_confidence", "ocr_confidence", "rule_code", "rule_label", "image_path", "crop_path", "metadata_path"]


class RecordStore:
    def __init__(self, directory=DATA_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.directory / "detections.csv"
        self.lock = threading.RLock()

    def records(self):
        with self.lock:
            if not self.csv_path.exists():
                return []
            with self.csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as file:
                reader = csv.DictReader(file)
                records = []
                seen = set()
                for row in reader:
                    # DictReader fills missing cells with None. Keep malformed
                    # rows out of every consumer, including history and rules.
                    if (None in row or any(row.get(field) is None for field in FIELDS)
                            or any(not row[field].strip() for field in ("id", "detected_at", "plate"))
                            or any("\ufffd" in value for value in row.values() if isinstance(value, str))):
                        logging.warning("Skipping incomplete CSV record at %s:%s", self.csv_path, reader.line_num)
                        continue
                    try:
                        scores = [float(row[key]) for key in ("ocr_confidence", "detector_confidence")]
                        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
                            raise ValueError("Invalid confidence")
                        seconds = row.get("video_seconds")
                        if seconds and (not math.isfinite(float(seconds)) or float(seconds) < 0):
                            raise ValueError("Invalid video position")
                        if row["id"] in seen:
                            raise ValueError("Duplicate record id")
                    except (TypeError, ValueError):
                        logging.warning("Skipping invalid CSV record at %s:%s", self.csv_path, reader.line_num)
                        continue
                    seen.add(row["id"])
                    records.append(row)
                return records

    def query(self, plate="", day="", restricted_only=False):
        result = self.records()
        if plate:
            result = [r for r in result if plate in r.get("plate", "")]
        if day:
            result = [r for r in result if r.get("detected_at", "").startswith(day)]
        if restricted_only:
            result = [r for r in result if r.get("rule_code") == "restricted"]
        return sorted(result, key=lambda r: r["detected_at"], reverse=True)

    def resolve(self, relative):
        root = self.directory.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Бүртгэлийн файл хадгалалтын хавтас дотор байх ёстой.")
        return target

    def add(self, record, annotated_rgb, crop_rgb, rule, at=None):
        from PIL import Image
        record = dict(record)
        record["plate"] = normalize_plate(record.get("plate", ""))
        if not record["plate"]:
            raise ValueError("Хадгалах дугаар 4 цифр, 3 үсэгтэй бүрэн дугаар байх ёстой.")
        at = at or now_ub()
        token = at.strftime("%H%M%S_%f") + "_" + uuid4().hex[:6]
        relative = Path("evidence") / at.date().isoformat()
        folder = self.directory / relative
        folder.mkdir(parents=True, exist_ok=True)
        stem = token + "_" + record["plate"]
        image_path = relative / (stem + ".jpg")
        crop_path = relative / (stem + "_plate.jpg")
        metadata_path = relative / (stem + ".json")
        row = {k: "" for k in FIELDS}
        row.update(record)
        row.update(id=token, detected_at=at.isoformat(timespec="milliseconds"), rule_code=rule.code, rule_label=rule.label,
                   image_path=image_path.as_posix(), crop_path=crop_path.as_posix(), metadata_path=metadata_path.as_posix())
        created = []
        try:
            for relative_path, pixels, quality in [(image_path, annotated_rgb, 94), (crop_path, crop_rgb, 97)]:
                target = self.resolve(relative_path)
                created.append(target)
                Image.fromarray(pixels).save(target, quality=quality)
            target = self.resolve(metadata_path)
            created.append(target)
            atomic_json(target, {**row, "rule": rule.as_dict(), "rule_basis": "user_configured_schedule", "time_basis": "import_processing_time" if row["source_kind"] != "camera" else "live_capture_time"})
            with self.lock:
                previous_size = self.csv_path.stat().st_size if self.csv_path.exists() else 0
                try:
                    with self.csv_path.open("a", encoding="utf-8-sig", newline="") as file:
                        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
                        if not previous_size:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception:
                    try:
                        if self.csv_path.exists():
                            with self.csv_path.open("r+b") as file:
                                file.truncate(previous_size)
                    except OSError:
                        logging.exception("Could not roll back incomplete CSV append")
                    raise
        except Exception:
            for path in created:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logging.exception("Could not clean up an incomplete evidence file")
            raise
        return row

    def export(self, destination, records):
        destination = Path(destination).resolve()
        if destination == self.csv_path.resolve():
            raise ValueError("Экспортын файлыг үндсэн detections.csv-ээс өөр нэрээр хадгална уу.")
        with destination.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({k: "'" + str(v) if str(v).startswith(("=", "+", "-", "@")) else v for k, v in record.items()})
