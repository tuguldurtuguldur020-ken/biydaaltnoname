# License plate detector

`plate_detector.pt` comes from
[Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection).
The model card declares MIT; Ultralytics and EasyOCR retain their own licenses.

- YOLOv8n, one class `license_plate`.
- Revision: `83c98fbe7412fe8b3950adb5637cfd08b0f04809`
- SHA-256: `2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822`
- Size: 6,248,291 bytes.

It has not been fine-tuned or evaluated on a Mongolian dataset here.
Use `python -m tools.train` with labelled Mongolian plates and select `mongolian_best.pt` in Settings.
OCR weights are cached under `data/models/ocr` after initial download.
