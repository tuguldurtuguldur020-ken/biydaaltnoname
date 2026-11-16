"""Capture starts immediately; AI runs independently of the Tk event loop."""
from copy import deepcopy
import logging
from pathlib import Path
import queue
import threading
import time
from .media import open_camera, open_video, read_image, rotate_frame
from .plates import PlateTracker, TemporalGate
from .recognition import PlateRecognizer, annotate
from .rules import evaluate
from .settings import now_ub


class DetectionWorker:
    def __init__(self, settings, store):
        self.settings = deepcopy(settings)
        self.store = store
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.previews = queue.Queue(maxsize=1)
        self.review_frames = queue.Queue(maxsize=1)
        self.inference_lock = threading.RLock()
        self.frame_worker = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.camera_ready = threading.Event()
        self.thread = None
        self.capture_thread = None
        self.playback_done = threading.Event()
        self.analysis_failed = False
        self.media_failed = False
        self.rotation = 0
        self.video_review = False
        self.video_fps = 25.0
        self.recognizer = None
        self.latest = None
        self.actual_camera_index = None
        self.latest_lock = threading.Lock()
        self.processed = 0
        self.saved_count = 0
        self.tracker = PlateTracker()

    @property
    def alive(self):
        return (any(thread is not None and thread.is_alive() for thread in (self.thread, self.capture_thread))
                or bool(self.frame_worker and self.frame_worker.alive))

    def stop(self):
        self.stop_event.set()
        self.pause_event.clear()
        if self.frame_worker:
            self.frame_worker.stop()

    def emit(self, kind, value):
        self.events.put((kind, value))

    def publish(self, frame, source=None, seconds=None):
        self.put_latest(self.frames, frame)
        self.put_latest(self.previews, (frame, source, seconds))

    @staticmethod
    def put_latest(target, value):
        try:
            target.put_nowait(value)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(value)
            except queue.Full:
                pass

    def start(self, kind, source):
        if self.alive:
            return False
        self.stop_event.clear()
        self.pause_event.clear()
        self.camera_ready.clear()
        self.playback_done.clear()
        self.analysis_failed = False
        self.media_failed = False
        self.rotation = int(self.settings.get("_rotation", 0)) % 360
        self.video_review = bool(self.settings.get("_video_review", False))
        self.frame_worker = None
        while not self.review_frames.empty():
            self.review_frames.get_nowait()
        self.capture_thread = None
        while not self.events.empty():
            try:
                self.events.get_nowait()
            except queue.Empty:
                break
        while not self.frames.empty():
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        while not self.previews.empty():
            self.previews.get_nowait()
        self.latest = None
        self.processed = self.saved_count = 0
        self.tracker = PlateTracker()
        self.thread = threading.Thread(target=self.run, args=(kind, source), daemon=True, name="platewatch-inference")
        self.thread.start()
        return True

    def wait_if_paused(self):
        while self.pause_event.is_set() and not self.stop_event.wait(.08):
            pass
        return self.stop_event.is_set()

    def load_recognizer(self):
        with self.inference_lock:
            self._load_recognizer()

    def _load_recognizer(self):
        if self.stop_event.is_set():
            raise InterruptedError()
        if self.recognizer is None:
            self.recognizer = PlateRecognizer(self.settings, lambda text: self.emit("status", text), self.stop_event)
        if self.stop_event.is_set():
            raise InterruptedError()

    def check_frame(self, rgb, kind, source, seconds=0):
        """Check the exact frozen pixels; share a serialized model, not its stop flag."""
        if self.frame_worker and self.frame_worker.alive:
            return False
        manual = DetectionWorker(self.settings, self.store)
        manual.inference_lock = self.inference_lock
        manual.video_fps = self.video_fps
        manual.events = self.events
        self.frame_worker = manual
        manual.thread = threading.Thread(target=manual.run_frame_check,
                                         args=(self, rgb.copy(), kind, source, seconds),
                                         daemon=True, name="platewatch-frozen-frame")
        manual.thread.start()
        return True

    def run_frame_check(self, owner, rgb, kind, source, seconds):
        import cv2
        try:
            with self.inference_lock:
                self.recognizer = owner.recognizer
                self.load_recognizer()
                if owner.recognizer is None:
                    owner.recognizer = self.recognizer
                self.gate = TemporalGate(1, float(self.settings["cooldown_seconds"]))
                compare_video = (kind == "video" and Path(source).is_file()
                                 and int(self.settings.get("neighbor_frames", 2)) > 0)
                detections, annotated = self.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), kind, source,
                                                      seconds, single_image=not compare_video, detailed=True)
                requested_rgb, requested_seconds = rgb, seconds
                from_neighbor = False
                if (compare_video and (not detections or any(not d.plate for d in detections))
                        and not self.stop_event.is_set()):
                    # Compression blur differs between adjacent *real* frames.
                    # Require two matching readings before replacing an unreadable
                    # frozen frame with a nearby, legible frame.
                    self.emit("frame_status", "Машин тус бүрийн дугаарыг ойролцоох кадруудаар баталгаажуулж байна...")
                    capture = None
                    try:
                        capture, _, info = open_video(source, self.stop_event)
                        limit = int(self.settings.get("neighbor_frames", 2))
                        for step in range(1, limit + 1):
                            offset = step / info["fps"]
                            if self.stop_event.is_set():
                                raise InterruptedError()
                            target = max(0, requested_seconds + offset)
                            if info["duration"] and target >= info["duration"]:
                                continue
                            capture.set(cv2.CAP_PROP_POS_FRAMES, round(target * info["fps"]))
                            ok, neighbor = capture.read()
                            if not ok or neighbor is None:
                                continue
                            neighbor = rotate_frame(neighbor, owner.rotation)
                            items, image = self.process(neighbor, kind, source, target, detailed="video")
                            if (sum(bool(d.plate) for d in items) > sum(bool(d.plate) for d in detections)
                                    or (any(d.plate for d in items) and not any(d.plate for d in detections))):
                                detections, annotated = items, image
                                rgb, seconds = cv2.cvtColor(neighbor, cv2.COLOR_BGR2RGB), target
                                from_neighbor = True
                            if items and all(d.plate for d in items):
                                break
                    finally:
                        if capture is not None:
                            capture.release()
                if not self.stop_event.is_set():
                    self.emit("frame_result", {"rgb": annotated, "original": rgb, "seconds": seconds,
                                                "requested_original": requested_rgb, "requested_seconds": requested_seconds,
                                                "from_neighbor": from_neighbor, "detections": detections, "saved": self.saved_count})
        except InterruptedError:
            pass
        except Exception as exc:
            logging.exception("Frozen frame check failed")
            self.emit("frame_error", str(exc))
        finally:
            self.stop_event.set()
            self.emit("frame_check_done", None)

    def run(self, kind, source):
        logging.info("Source start: %s %s", kind, source)
        self.gate = TemporalGate(1, float(self.settings["cooldown_seconds"]))
        try:
            if kind == "image":
                self.run_images(source)
            elif kind == "camera":
                self.run_camera(source)
            elif kind == "video":
                self.run_video(source)
            else:
                raise RuntimeError(f"Эх үүсвэрийн төрөл буруу: {kind}")
            if not self.stop_event.is_set() and not self.analysis_failed and not self.media_failed:
                text = f"Дууслаа • {self.processed} кадр шалгаж, {self.saved_count} дугаар хадгаллаа."
                if not self.saved_count:
                    text += " Уншигдахуйц Монгол дугаар илрээгүй."
                self.emit("status", text)
        except InterruptedError:
            pass
        except Exception as exc:
            logging.exception("Detection failed: %s", source)
            self.emit("error", str(exc))
        finally:
            self.stop_event.set()
            self.emit("done", None)
            logging.info("Source finished: %s; saved=%s", kind, self.saved_count)

    def run_images(self, source):
        import cv2
        paths = [source] if isinstance(source, (str, Path)) else list(source)
        if not paths:
            return
        for index, path in enumerate(paths):
            if self.wait_if_paused():
                return
            frame = rotate_frame(read_image(path), self.rotation)
            # A selected picture is visible even if model loading is slow or fails.
            self.publish(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), source=str(path), seconds=0)
            self.emit("source_ready", {"kind": "image", "source": str(path)})
            self.emit("status", f"Зураг нээгдлээ • {Path(path).name} • AI бэлтгэж байна...")
            self.load_recognizer()
            self.emit("ready", "image")
            _, annotated = self.process(frame, "image", str(path), 0, single_image=True)
            self.put_latest(self.review_frames, (annotated, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0, str(path)))
            self.emit("progress", (index + 1) / len(paths))

    def run_camera(self, index):
        camera_thread = threading.Thread(target=self.capture_camera, args=(index,), daemon=True, name="platewatch-camera")
        self.capture_thread = camera_thread
        camera_thread.start()
        try:
            while not self.camera_ready.wait(.05):
                if self.stop_event.is_set() or not camera_thread.is_alive():
                    return
            if self.stop_event.is_set():
                return
            # Capture is already publishing frames while the heavy models load.
            try:
                self.load_recognizer()
                self.emit("ready", "camera")
                seen = -1
                while not self.stop_event.wait(.02):
                    if self.wait_if_paused():
                        return
                    with self.latest_lock:
                        sample = self.latest
                    if sample is None or sample[0] == seen:
                        continue
                    seen, frame, captured_at = sample
                    self.process(frame, "camera", f"Камер {self.actual_camera_index}", 0, at=captured_at)
            except InterruptedError:
                raise
            except Exception as exc:
                self.analysis_error(exc)
                # Camera preview remains usable when model loading/inference fails.
                while camera_thread.is_alive() and not self.stop_event.wait(.1):
                    pass
        finally:
            self.stop_event.set()
            camera_thread.join(timeout=3)

    def capture_camera(self, index):
        import cv2
        capture = None
        try:
            capture, frame, backend, actual_index = open_camera(index, self.settings.get("camera_backend", "auto"),
                                                  lambda text: self.emit("status", text), self.stop_event)
            self.actual_camera_index = actual_index
            frame = rotate_frame(frame, self.rotation)
            self.emit("source_ready", {"kind": "camera", "source": str(actual_index), "backend": backend, "shape": list(frame.shape)})
            failures = 0
            number = 0
            while not self.stop_event.is_set():
                if self.wait_if_paused():
                    break
                number += 1
                with self.latest_lock:
                    self.latest = (number, frame, now_ub())
                self.publish(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.camera_ready.set()
                ok, next_frame = capture.read()
                if not ok or next_frame is None or not next_frame.size:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Камерын дүрс тасарлаа. Камераа дахин эхлүүлнэ үү.")
                    self.stop_event.wait(.1)
                    continue
                failures = 0
                frame = rotate_frame(next_frame, self.rotation)
        except InterruptedError:
            pass
        except Exception as exc:
            logging.exception("Camera failed")
            self.emit("error", str(exc))
            self.stop_event.set()
        finally:
            if capture is not None:
                capture.release()
            self.camera_ready.set()

    def analysis_error(self, exc):
        logging.exception("AI analysis failed; retaining media preview")
        self.analysis_failed = True
        self.recognizer = None
        self.emit("analysis_error", "AI шалгалт зогслоо; дүрс үргэлжилнэ. " + str(exc))

    def play_video(self, capture, frame, info):
        """Real-time playback owns its decoder; inference never blocks it."""
        import cv2
        index = 0
        try:
            while not self.stop_event.is_set():
                if self.wait_if_paused():
                    return
                started = time.monotonic()
                self.publish(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), seconds=index / info["fps"])
                index += 1
                self.emit("playback", (index / info["fps"], info["duration"]))
                speed = max(.25, min(4, float(self.settings.get("playback_speed", 1))))
                self.stop_event.wait(max(0, 1 / (info["fps"] * speed) - (time.monotonic() - started)))
                if self.stop_event.is_set():
                    return
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    if info["total"] and index < info["total"] - 2:
                        raise RuntimeError(f"Видеоны дүрс {index / info['fps']:.1f} секунд дээр тасарлаа. Файлаа шалгана уу.")
                    break
                frame = rotate_frame(frame, self.rotation)
        except Exception as exc:
            logging.exception("Video playback failed")
            self.media_failed = True
            self.emit("error", str(exc))
        finally:
            capture.release()
            self.playback_done.set()
            self.emit("playback_done", self.media_failed)

    def run_video(self, source):
        import cv2
        playback, first_frame, info = open_video(source, self.stop_event)
        first_frame = rotate_frame(first_frame, self.rotation)
        info["shape"] = list(first_frame.shape)
        self.emit("source_ready", {"kind": "video", "source": str(source), **info})
        self.publish(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), source=str(source), seconds=0)
        if self.video_review:
            self.playback_done.set()
        else:
            self.capture_thread = threading.Thread(target=self.play_video, args=(playback, first_frame, info),
                                                   daemon=True, name="platewatch-video-playback")
            self.capture_thread.start()
        capture = None
        recorder = None
        try:
            self.load_recognizer()
            self.emit("ready", "video")
            # A second decoder checks EVERY configured sample from the start.
            # Queueing all full-size frames or dropping slow samples would either
            # exhaust memory or miss plates while the models are loading.
            if self.video_review:
                capture, frame = playback, first_frame
            else:
                capture, frame, _ = open_video(source, self.stop_event)
                frame = rotate_frame(frame, self.rotation)
            fps, total = info["fps"], info["total"]
            self.video_fps = fps
            if self.settings.get("save_video"):
                folder = self.store.directory / "videos"
                folder.mkdir(exist_ok=True)
                path = folder / (now_ub().strftime("%Y%m%d_%H%M%S_%f") + ".mp4")
                recorder = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                if not recorder.isOpened():
                    raise RuntimeError("Видео хадгалах codec ажилласангүй. Тохиргооноос видео хадгалахыг унтраана уу.")
            index = 0
            last_sample = None
            detections = []
            while not self.stop_event.is_set():
                # Pausing a video freezes its view; analysis continues from the
                # beginning and the user can independently inspect that frame.
                stride = max(1, int(self.settings["frame_stride"]))
                if self.settings.get("adaptive_sampling", True) and any(
                        t.get("pending") and 0 <= index / fps - t["time"] <= 1 for t in self.tracker.tracks):
                    # A readable view may last less than the configured gap.
                    # Confirm a seen number before returning to normal sampling.
                    stride = min(stride, max(1, round(fps * .1)))
                if last_sample is None or index - last_sample >= stride:
                    detections, rgb = self.process(frame, "video", str(source), index / fps,
                                                   detailed="video")
                    last_sample = index
                    if self.video_review:
                        self.put_latest(self.review_frames, (rgb, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index / fps))
                    self.emit("analysis_position", (index / fps, info["duration"]))
                elif recorder:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if recorder:
                    recorder.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                index += 1
                if index % stride == 0 and total:
                    self.emit("progress", min(1.0, index / total))
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    if total and index < total - 2:
                        raise RuntimeError(f"AI шалгалт {index / fps:.1f} секунд дээр видео уншиж чадсангүй.")
                    break
                frame = rotate_frame(frame, self.rotation)
            if not self.stop_event.is_set():
                self.emit("progress", 1.0)
                self.emit("analysis_position", (index / fps, info["duration"]))
                self.emit("status", f"AI шалгалт дууслаа • {self.processed} кадр • {self.saved_count} дугаар хадгаллаа.")
        except InterruptedError:
            raise
        except Exception as exc:
            self.analysis_error(exc)
        finally:
            if capture is not None:
                capture.release()
            if recorder:
                recorder.release()
            while not self.playback_done.wait(.05):
                if self.stop_event.is_set():
                    break
            if self.video_review:
                playback.release()
            elif self.capture_thread:
                self.capture_thread.join(timeout=3)

    def process(self, frame, kind, source, video_seconds, single_image=False, at=None, detailed=False):
        if self.frame_worker and self.frame_worker.alive:
            self.frame_worker.thread.join()
        with self.inference_lock:
            return self._process(frame, kind, source, video_seconds, single_image, at, detailed)

    def _process(self, frame, kind, source, video_seconds, single_image=False, at=None, detailed=False):
        import cv2
        started = time.monotonic()
        at = at or now_ub()
        method = ("detect_video_frame" if detailed == "video" else "detect_detailed" if detailed
                  else "detect_image" if kind == "image" else "detect")
        if kind == "camera" and self.settings.get("recognition_mode") == "detailed":
            method = "detect_detailed"
        detect = getattr(self.recognizer, method, self.recognizer.detect)
        has_cancel = hasattr(self.recognizer, "cancel")
        previous_cancel = getattr(self.recognizer, "cancel", None)
        if has_cancel:
            self.recognizer.cancel = self.stop_event
        try:
            # A manually checked frame shares this model. Runtime controls apply
            # at inference boundaries while the same lock protects all calls.
            if hasattr(self.recognizer, "settings"):
                self.recognizer.settings = self.settings
            detections = detect(frame)
        finally:
            if has_cancel:
                self.recognizer.cancel = previous_cancel
        if self.stop_event.is_set():
            return detections, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clock = video_seconds if kind == "video" else time.monotonic()
        if not single_image and kind in ("video", "camera"):
            window = (max(3, 1.5 * int(self.settings["frame_stride"]) / self.video_fps) if kind == "video"
                      else max(3, 2 * (time.monotonic() - started)))
            detections = self.tracker.confirm(detections, clock,
                                              int(self.settings["confirmation_frames"]),
                                              float(self.settings.get("video_min_confidence", .65)), window=window)
        self.processed += 1
        if kind == "camera":
            # Let the GUI draw recent boxes over the newest camera frame.
            self.emit("overlay", (detections, time.monotonic()))
        rgb = annotate(frame, detections, self.settings["rules"], at)
        seen = set()
        for detection in detections:
            plate = detection.plate
            if not plate or plate in seen:
                continue
            seen.add(plate)
            rule = evaluate(plate, self.settings["rules"], at)
            self.emit("detection", (detection, rule))
            if not self.gate.ready(plate, clock, single_image=single_image):
                continue
            x1, y1, x2, y2 = detection.bbox
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            record = self.store.add({"plate": plate, "source": source, "source_kind": kind,
                                     "video_seconds": round(video_seconds, 3) if kind == "video" else "",
                                     "raw_ocr_text": detection.raw_text, "bbox": list(detection.bbox),
                                     "method": getattr(detection, "method", "yolo_ocr"),
                                     "reading_support": getattr(detection, "reading_support", 1),
                                     "detector_confidence": round(detection.detector_confidence, 4),
                                     "ocr_confidence": round(detection.ocr_confidence, 4)}, rgb, crop, rule, at)
            self.gate.mark_saved(plate, clock)
            self.saved_count += 1
            self.emit("record", record)
        elapsed = time.monotonic() - started
        self.emit("metrics", (len(detections), round(elapsed * 1000), round(1 / max(.001, elapsed), 1)))
        if kind == "image":
            self.publish(rgb, source=source, seconds=0)
        return detections, rgb
