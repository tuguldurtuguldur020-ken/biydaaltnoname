# PlateWatch 2.8.1 validation

## Folder organization check

Maintenance/verification scripts now live in `tools/`, guides in `docs/`, and the dataset template in `configs/`. Run tools from the project root as `python -m tools.<name>`. The previous complete README is preserved as `docs/USER_GUIDE_MN.md`; move paths are recorded in `archive/reorganization_manifest.json`.

After organization, all **109 tests passed** in 29.111 seconds; all 12 tool modules imported, and all 10 argument-based tool help commands exited successfully. The moved `tools.verify_project` also passed actual `7960УЕХ` image recognition, GUI manual recheck, three negative images, history/export and settings checks. `run.bat --smoke-test` opened and closed successfully. [Reorganization unit-test log](../test_artifacts/reorganization/unit_tests.log), [actual moved-tool report](../test_artifacts/reorganization/actual/report.json). A new EXE build was not performed.

## Recognition validation

Validated on 2026-11-17 against the newly reported original photo, not its screenshot. The source is `846220724_1123758723635137_4283606675668681012_n.jpg`; its plate is **7960УЕХ**. The previous 2.8.0 validation did not include this image.

## Newly reproduced failure and correction

The old 640-pixel full-frame pass returned no boxes. A tile found an oversized box containing the bumper; fixed horizontal row splits and the text-size filters then failed on the slanted, smaller letter row. The actual baseline report contains an empty plate and raw OCR `E`. [Baseline](../test_artifacts/audit_2_8_1/baseline.json).

Added `platewatch/layout.py` to locate actual character components and aligned number/letter rows. Local contrast and bounded threshold fallbacks separate text from bumper trim; each detected row is straightened from its own geometry. Segmentation uses a temporary mask, while OCR reads the original color/grayscale pixels. Missing characters are never added. Two preprocessing views must produce the same complete plate; conflicting or incomplete readings remain rejected. A confirmed detailed primary reading is retained when the secondary reader produces no complete plate. Static detection also retries at 960 pixels when the initial pass finds no box in a large image.

The original photo now reads **7960УЕХ**, both during ordinary import and through the actual GUI's manual frame-check button. It took **1.40 seconds with models loaded** in the combined run; cold model startup is additional. GUI scores were OCR 70% / YOLO 40%; these are model scores, not an evaluated correctness probability. [Corrected diagnostic](../test_artifacts/audit_2_8_1/fixed.json), [actual GUI screenshot](../test_artifacts/audit_2_8_1/final/reported_photo_gui.png).

## Final checks

- **109 automated tests passed** in 28.342 seconds. Added tests cover both slant directions, oversized crops, touching-character segmentation, missing digits, empty panels, matching/conflicting row readings and preserving a valid primary result. [Unit-test log](../test_artifacts/audit_2_8_1/unit_tests.log).
- **10 positive image cases passed**: nine user photos, including the newly reported one, and one generated fixture. The earlier nine image cases all retain their expected plates. Six real negative cases (a book and five unsupported foreign formats) and three generated negative cases produced no accepted plate. The verifier now requires exact expected plate sets, so extra false readings fail as well as missing readings. Labels/paths: [manifest](../test_artifacts/audit_2_8_1/cases.json).
- The full 367-frame user video passed in balanced mode: 64 adaptive samples, exactly **7034УБУ**, **6657УНА**, **4000УЕО**, in 44.81 seconds with models already loaded. Detailed mode also passed with exactly the same three plates: 69 adaptive samples in 151.45 seconds with loaded models. [Detailed-mode report](../test_artifacts/audit_2_8_1/detailed/report.json). Detailed inspection remains substantially slower than balanced inspection.
- A physical 640×480 camera produced previews and completed three actual inference cycles, then stopped, released its device and reopened without an error.
- The actual Tk GUI passed reported-photo import, manual recheck, history filtering, filtered CSV export, restriction rules and settings save/reload. Verification records were isolated from production records.
- `run.bat --smoke-test` opened and closed successfully. Source version is **2.8.1**; close the old app before launching `run.bat`. The existing EXE was not rebuilt. Main-screen controls remain uncluttered, with speed/stride/mode under Settings.

[Combined acceptance report](../test_artifacts/audit_2_8_1/final/report.json). Repeat with `python -m tools.verify_project --manifest test_artifacts/audit_2_8_1/cases.json --output test_artifacts/audit_2_8_1/recheck`; add `--camera` for the physical device and `--video <path>` for the known user clip. `--mode detailed` selects the detailed mode.

These checks cover this regression and the listed fixtures, not all Mongolian road conditions. The small pickup in the video remains unconfirmed. Unreadable characters, unsupported plate formats and unseen lighting/angle conditions still require independent evaluation; expected strings and source filenames are not used by production recognition code.

## Earlier 2.8.0 validation

Validated on 2026-11-17 with this project's Python 3.13 virtual environment, actual YOLO/PP-OCRv5/EasyOCR weights, the user's original images and video, and a physical camera.

## Recognition and evidence

The reported square plate **1742УБР** reproduced the failure: the previous detector returned no boxes. Added a bounded geometric search for a light quadrilateral containing character shapes, perspective rectification, and complete OCR agreement before acceptance. Geometric candidates have no YOLO score; the GUI displays `YOLO —`, and metadata records `geometry_ocr`. No expected plate strings or filename rules were added to recognition code.

Image imports and the diagnostic CLI use the full image inspection path. Text groups around an emblem are compared by both recognizers. Agreement is preserved when merging overlapping boxes, preventing a later, more confident single-model reading from replacing it. Greek visual lookalikes emitted by OCR are normalized before plate validation.

All **9 image cases** passed with matching full-frame JPEG, plate crop, JSON metadata and CSV records:

| Source type | Expected and recorded plate |
| --- | --- |
| User photo, reported failure | 1742УБР |
| User photo | 3892УНУ |
| User photo | 8804ДОУ |
| User photo | 7319ДАР |
| User photo | 3225ЗАА |
| User photo | 9638УЕН |
| User photo | 1771УБН |
| User photo | 0878УНК |
| Generated fixture | 1234УБА |

The final image/GUI rerun also passed three negative cases through `detect_image`: an empty frame, an empty light panel and ordinary text. None produced a plate. Source paths and labels are in [cases.json](../test_artifacts/audit_2_8/cases.json). Reports: [combined acceptance](../test_artifacts/audit_2_8/final/report.json), [final image and GUI rerun](../test_artifacts/audit_2_8/final_images/report.json). See the [actual GUI screenshot](../test_artifacts/audit_2_8/final_images/reported_photo_gui.png).

The full 367-frame, 30 FPS user video passed: 64 adaptive samples recorded **7034УБУ**, **6657УНА**, **4000УЕО**, with the known wrong variants absent. This run took 94.09 seconds with models already loaded and competing unit-test work; it is not a standalone speed benchmark. The small pickup's suffix remains unconfirmed. Detailed mode and close-frame confirmation trade processing time for additional evidence; speed controls remain in Settings.

The physical camera supplied 640×480 previews, completed three real inference cycles, stopped and released its device, then reopened successfully. Camera frames without accepted plates were not saved by the verifier.

## Application and recovery

- All **103 automated tests** passed in 28.188 seconds. Coverage includes OCR conflicts, square plates, bounded geometry, tracking across configured sample gaps, preview recovery, import switching, pause/resume, rotation, evidence, rules and settings. Injected model/IO failures in the log are expected test scenarios. [Unit-test log](../test_artifacts/audit_2_8/unit_tests_utf8.log).
- Actual Tk GUI checks passed for the reported photo, manual recheck, history, filtered CSV export, configured restriction schedule, and settings save/reload. An unpadded date filter is normalized before lookup. Changing ordinary settings retains loaded models and the source rotation; changing the model/device resets them.
- Preview packets carry their own source and timestamp. A batch image recheck uses the displayed image, and a video recheck uses the exact displayed frame position. Unsampled recorded video frames do not inherit boxes from earlier frames.
- Damaged settings values recover independently to valid defaults; the original settings file is preserved. JSON uses unique temporary files and serialized atomic replacement. Invalid/duplicate CSV rows are skipped without rewriting the original. Failed evidence/CSV writes clean up incomplete files and roll back a partial append. Record paths are confined to the evidence directory. Missing evidence and export failures are handled in the GUI.
- `run.bat --smoke-test` opened and closed successfully on the updated source. Version **2.8.0** appears in the GUI. The existing 2.2.0 EXE was not rebuilt; close the old window and launch the root `run.bat`.

Run `python -m tools.verify_project --manifest test_artifacts/audit_2_8/cases.json --output test_artifacts/audit_2_8/recheck` to repeat image/GUI checks in an isolated records directory. Manifest paths point to this machine's originals. Add `--camera` for the physical device; `--video <path>` specifically checks the three known plates in the reported user clip. Original source images and production records were not rewritten.

These are regression checks on eight real stills, one generated still, three negative frames and one clip. They are not an independent Mongolian road-camera accuracy evaluation. Blurred/occluded characters and other plate formats can still remain unreadable; OCR confidence is a model score rather than a measured probability of correctness.

## Earlier 2.7.1 UI update

Removed the mode, frame-stride, playback-speed and Apply controls from the monitor. They now appear together under Settings; live application and the main settings-save action both preserve them. The preview gains the space occupied by the old controls row. Recognition code is unchanged. Screenshots: `test_artifacts/monitor_clean_2_7_1.png` and `settings_video_2_7_1.png`.

## Earlier 2.7.0 validation

All **85 automated tests** passed on the final source. Added coverage includes CTC repeated digits, weak final-character scores, conflicting recognizers, physical plate crops, square-plate rows, checksum rejection, adaptive sampling, and multi-car manual confirmation.

Validated on 2026-11-16 using Python 3.13, the actual user MP4 and its reported frozen frame.

- Added the official PaddlePaddle Cyrillic PP-OCRv5 mobile ONNX recognizer (8 MB), pinned to revision `2cef88145434beb8afa9dd82d77d799eb1ad7b29`. Both weights and the matching character dictionary are verified with SHA-256. ONNX Runtime runs locally on CPU. The older Mongolian EasyOCR path remains available for readings the new model cannot establish.
- Corrected the reported `6657УНХ`/`6657УНЬ` misread: the real plate is read as `6657УНА`. The actual frozen frame at source time 106/30 seconds was checked with its neighbors; the result at 3.5667 seconds preserved the matching native frame, bbox and crop. The other plate in that view remained explicitly unconfirmed instead of being counted as a fully read number.
- The complete video regression now checks the later cars as well as `7034УБУ`: `6657УНА` and the two-row `4000УЕО` must also be recorded, and the known wrong `6657УНХ`, `6657УНЬ` and `1703УБУ` must be absent. Real GUI screenshots and reports are in `test_artifacts/accuracy_2_7/`. Repeat with `python -m tools.verify_multicar --video <source.mp4> --mode balanced` or `--mode detailed`.
- Final balanced GUI verification passed: 64 adaptive samples, all three expected plate records, 48.58 seconds including model startup. Detailed verification also passed: 53 adaptive samples, the same three records, 143.91 seconds including model startup. Detailed mode trades time for the higher resolution/tile search; it is not advertised as faster than balanced mode.
- Two-row plates are read as real number and letter rows. The previous blanket 45-pixel width cutoff no longer discards readable square plates. Detailed review uses a higher detector input resolution, with a native-pixel tile search when the frame cannot be read. Balanced review retains the smaller detector input for speed.
- Default adaptive sampling temporarily checks closer frames when a number has appeared but still needs confirmation. It returns to the configured stride afterward and can be disabled in Settings. Manual video inspection also compares partially readable multi-car scenes; one recognized car no longer prevents checking the others. The first, unconfirmed frame is not saved immediately.
- Model score selection includes weak character scores; a doubtful final letter cannot hide behind confident digits. Conflicting independent recognizers are rejected rather than selecting whichever model reports the larger score. No guessed plate strings or source-specific substitutions were added to production code.
- Actual image regressions passed for the previous Toyota `9638УЕН`, user photo `3892УНУ`, and synthetic car `1234УБА`; see `image_regressions.json`. The small pickup's suffix is still below confirmation thresholds in the source clip. Registration/averaging experiments were diagnostic only and were not shipped as a plate-reading shortcut.
- The source version is **2.7.0**. Launch with `run.bat`; its dependency check now includes ONNX Runtime and PyYAML. The existing EXE was not rebuilt. This is a regression check, not an evaluated general road-camera accuracy score.

## Earlier 2.6.0 checks

Validated on 2026-11-16 using the project `.venv` Python environment.

- All **74 automated tests** pass. Added regressions cover conflicting readings on the same car, separate cars with identical text, repeated processing of a source frame, an approaching vehicle moving between sampled frames, agreement between OCR variants, review sampling, runtime stride changes, and applying/persisting controls during playback without reloading the model.
- Reproduced the reported error on the user's actual video: the old detailed path returned `1703УБУ` from the frame at 1.17 seconds and took 7.98 seconds. The corrected balanced path returned `7034УБУ` in 0.98 seconds on the same extracted frame; the corrected detailed path also returned `7034УБУ`.
- Actual GUI + YOLO + EasyOCR checked the complete 367-frame, 30 FPS video (12.23 seconds), using a stride of 10 and balanced recognition. All 37 configured samples completed in **29.3 seconds including model startup**. The first confirmed plate appeared after 12.17 seconds. Exactly one record, `7034УБУ` at source time 1.0 seconds, was saved. The previously reported `1034УБУ`, `1703УБУ` and `1032УБУ` were not recorded in this run.
- Evidence: `test_artifacts/accuracy_speed_2_6/report.json`, `confirmed_plate.png`, the isolated evidence records in that directory, and `test_artifacts/unit_tests_2_6.log`. Run `python -m tools.verify_accuracy_speed --video <source.mp4>` to repeat the GUI regression.
- Actual image regressions also passed with the same final OCR code: the reported frame in detailed mode (`7034УБУ`), the earlier Toyota photo (`9638УЕН`), the earlier user photo (`3892УНУ`), and the synthetic car (`1234УБА`). Results: `test_artifacts/accuracy_speed_2_6/image_regressions.json`.
- Review uses the configured stride; stride 1 checks every frame. The default for new settings is 10. Existing saved stride preferences are preserved. Larger gaps may miss brief readable views: at stride 15 this clip did not supply two readable samples of the first plate, so the stricter confirmation correctly produced no record. Runtime sampling is configurable to address this tradeoff.
- The main controls change OCR mode, sample stride and playback speed while the worker is active. Manual neighbor search is separately bounded to the configured count (default 2; 0 disables it). OCR confidence is evidence from the model, not a measured correctness probability. These checks establish the reported regression, not universal road-camera accuracy.
- Launch the updated source with `run.bat`. The earlier 2.2.0 EXE was not rebuilt.

## Earlier 2.4.0 checks

Validated on 2026-11-16 using the project Python 3.13 environment and OpenCV 5.0.

- All **51 automated tests** pass. Regressions cover playback during blocked model startup, checking every sampled video frame, camera/video preview surviving AI errors, decoder fallback, rejecting photos as videos, fullscreen/Escape, fitting/filling the canvas, and rotating/rechecking a source.
- The actual user MP4 in Downloads contains 351 frames at 30 FPS (11.7 seconds). Its pixels are sideways and its rotation metadata is zero. With the explicit 270-degree rotation, all **71 configured samples** were checked, playback reached **11.7 seconds**, and a plate record was saved. The first visible preview arrived in under 0.2 seconds, before model startup completed.
- A generated MP4 containing the existing synthetic car fixture was checked with actual YOLO and EasyOCR. All six sampled frames were processed and **1234УБА** was recorded. This is a pipeline check, not a measured road-camera accuracy score.
- The GUI pause/resume and fullscreen/Escape paths passed during actual playback. Fullscreen Escape left playback running.
- Physical camera 0 supplied 640×480 frames, completed three actual inference cycles, stopped, released its device and reopened successfully.
- Evidence: `test_artifacts/media_verification_2_4.json`, `video_monitor_2_4.png`, `video_fullscreen_2_4.png`, `video_recognition_2_4.png`, and the isolated `media_evidence_2_4` records.
- The source app is **2.4.0**. Launch with `run.bat`; the older EXE in `dist` was not rebuilt for this change.

Video playback and analysis use separate decoders. Playback retains only the newest frame; analysis decodes the file from the beginning and checks every configured `frame_stride`, rather than dropping samples when inference is slow or buffering a whole video in memory. Analysis and playback times are displayed separately. Model failures leave the media preview running with a visible error. Decoder failures are reported separately from successful completion.

The preview upscales and downscales while preserving aspect ratio, with a bounded Tk image and allocation recovery. Fill mode crops only the displayed view; recognition and saved evidence use the complete, explicitly rotated frame. Duplicate monitor history was removed to make room for the image; full history remains on its own page.

## Earlier 2.3.0 checks

The reported camera/import freeze was reproduced from the application log: at 10:55:51 on 2026-11-13, Pillow raised `MemoryError` while resizing a camera preview. The sole `poll()` callback scheduled its next invocation only on success, so it never ran again. Workers continued processing later selected files while the GUI stopped consuming frames, records and pending source changes.

The update loop now reschedules in `finally`. Preview rendering preserves the previous image on allocation failure and retries at a smaller resolution even when an imported photo has no further queued frames. Arrays are resized before Pillow/Tk conversion, preview dimensions are bounded, and a same-size Tk image is reused between camera frames. Pending history refreshes survive callback errors.

- All **39** automated tests passed with the project `.venv`, including injected preview `MemoryError`, an unrelated callback exception, image-only recovery without another frame, Tk image reuse, large-image preview and camera-to-image switching after an error.
- Real Windows/Tk end-to-end verification ran camera 0 (DSHOW, 640×480), deliberately failed the first camera preview allocation, observed automatic preview recovery, completed three actual YOLO/OCR camera inference cycles, then used the actual Windows file picker to import the user's Toyota photo.
- The imported photo produced **9638УЕН**, full-frame/crop JPEGs, CSV and JSON evidence. The recovered camera preview appeared after 2.121 seconds. `1113` still displayed Wednesday and the separately selected Friday remained allowed.
- Current check: `test_artifacts/native_import_verification.json`; screenshots: `native_import_gui.png` and `rules_1113_gui.png`.
- The existing EXE is the older 2.2.0 build; it does not contain this fix. Use the root `run.bat` / `camera.bat` to run 2.3.0. A future `build_exe.py` run takes its version directly from the Python package.

## Earlier 2.2.0 checks

- 35 automated tests passed using the project `.venv` Python 3.13.
- The user's exact Toyota photo, previously showing a `NoneType` error, was read as `9638УЕН` after retrying a wider plate crop and excluding the oval MGL emblem.
- The real Windows file picker was exercised through the new main-preview `+ Зураг` button. The Toyota image produced a saved image, crop, CSV row and metadata JSON.
- Typing `1113` now displays Wednesday as the configured restriction day. A separately selected Friday date remains an allowed date; this distinction was tested in both the GUI and rule engine.
- `test_artifacts/native_import_verification.json`, `toyota_photo_verification.json`, and `rules_1113_gui.png` record these checks.
- A physical-camera regression requested unavailable index 2 and automatically opened the Integrated Camera at index 0 through DSHOW. It delivered 291 preview frames, completed three inference cycles, stopped and reopened successfully.
- The user-selected real vehicle photo was processed after fixing an EasyOCR full-crop API error; the plate was read as `3892УНУ`.
- `camera.bat` and `biydaalt.py --camera` start the live camera directly. If the app is already open, a source request is sent to that window instead of opening a second instance.
- `run.bat --smoke-test` successfully opened and closed the updated GUI.
- Real camera index 0 opened with DSHOW at 640×480; the first preview arrived after 1.7 seconds. 304 preview frames and three inference cycles were observed without errors.
- Stopping the real camera released it; it was successfully reopened immediately afterward.
- A real GUI integration check started the physical camera, waited for AI inference, pressed the image import button, selected a fixture through the actual Windows file dialog, and switched from camera to image processing.
- Actual YOLO + Mongolian EasyOCR read `1234УБА` from the synthetic car fixture and saved full-frame JPEG, plate crop, JSON metadata, and CSV record in `test_artifacts/native_import_evidence`.
- `test_artifacts/camera_verification.json` and `test_artifacts/native_import_verification.json` record the physical camera and native dialog checks.
- `test_artifacts/python_verification_fixed.json` records a successful YOLO/OCR check after the memory fix.
- The tests cover button callbacks, cancelled dialogs, camera-to-image switching, Unicode filenames, TIFF, EXIF rotation, GUI responsiveness after AI errors, repeated readings, schedule/time boundaries, video import, and MP4 output.
- A Windows instance guard prevents repeated launches of this project from creating competing camera windows.

The live regression reproduced an OCR startup failure allocating 411,041,792 bytes for an unused VGG classifier. The updated code uses EasyOCR's documented `detector=False` and `recognize()` API on the YOLO crop, avoiding the second CRAFT/VGG detector. OpenCV finds prominent text rows when needed. Image/camera preview starts before AI loading, and errors are displayed in the GUI without blocking source changes.

PyInstaller successfully rebuilt `dist/PlateWatch/PlateWatch.exe` from the final 2.2.0 source, with Python, AI dependencies and local model weights. The resulting folder is approximately 672 MB. A launch test of this exact build failed with Windows error 4551: "An Application Control policy has blocked this file". Therefore the packaged GUI/AI runtime is not verified on this PC. The result is recorded in `test_artifacts/exe_verification_2_2.json`. No security policy was changed. The root `run.bat` is independently tested and the 2.2.0 Python GUI remains available.

No labelled Mongolian dataset was supplied, so no Mongolian fine-tuning or real-world recognition-accuracy evaluation was performed. The physical-camera checks establish capture, inference and release; the synthetic fixture establishes an actual recognition and evidence-saving path. They do not establish road-camera accuracy.
