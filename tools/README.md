# Туслах хэрэгслүүд

Эдгээрийг төслийн **үндсэн хавтаснаас** `.venv`-ийн Python-оор module хэлбэрээр ажиллуулна. Жишээ:

```powershell
.\.venv\Scripts\python.exe -m tools.verify_project --help
```

| Файл / module | Үүрэг |
| --- | --- |
| `setup_models.py` / `tools.setup_models` | YOLO, OCR жинг татаж/шалгаж бэлтгэх. Анх интернэт хэрэгтэй. |
| `train.py` / `tools.train` | Өөрийн тэмдэглэгээтэй YOLO өгөгдлөөр загвар нэмж сургах. |
| `build_exe.py` / `tools.build_exe` | PyInstaller-аар Windows EXE багц бүтээх; `build_exe.bat` үүнийг дуудна. |
| `verify_project.py` / `tools.verify_project` | Бодит загвар, зураг, GUI, хадгалалт, сонголтоор видео/камерын нэгдсэн шалгалт. |
| `verify_gui.py` / `tools.verify_gui` | Нэг зургийг бодит цонхоор таних, хадгалах шалгалт. |
| `verify_camera.py` / `tools.verify_camera` | Бодит камер, дүрс, inference, төхөөрөмж суллалтыг шалгах. |
| `verify_media.py` / `tools.verify_media` | Видео, үүсгэсэн бичлэг, дэлгэц, сонголтоор камерыг шалгах. |
| `verify_native_import.py` / `tools.verify_native_import` | Windows-ийн файл сонгох цонх, импорт, дүрс сэргэх ажиллагааг шалгах. |
| `verify_multicar.py` / `tools.verify_multicar` | Мэдээлсэн тодорхой бичлэгийн гурван машины дугаарыг шалгах. |
| `verify_accuracy_speed.py` / `tools.verify_accuracy_speed` | Өмнөх тодорхой бичлэгийн эхний дугаарын алдаа, хурдыг давтаж шалгах. |
| `verify_review.py` / `tools.verify_review` | Зогсоосон кадрын дахин шалгалтыг турших. |
| `verify_exe.py` / `tools.verify_exe` | Бүтээсэн EXE багцыг шалгах. Хуучин 2.2.0 багцын тайлангийн нэрийг ашигладаг. |

## Түгээмэл команд

```powershell
# Бодит зураг, GUI, хадгалалт шалгах
.\.venv\Scripts\python.exe -m tools.verify_project --manifest test_artifacts/audit_2_8_1/cases.json --output test_artifacts/recheck

# AI жин бэлтгэх
.\.venv\Scripts\python.exe -m tools.setup_models

# Өөрийн өгөгдлөөр сургах
.\.venv\Scripts\python.exe -m tools.train --data datasets/mongolian_plates/data.yaml --epochs 80 --device cpu
```

Manifest дахь эх зургийн замууд энэ компьютерийн файлуудыг заадаг. Өөр компьютер дээр зам, хүлээсэн дугаарыг өөрийн өгөгдөлд тааруулна. Нэгдсэн шалгалтад `--camera` нэмж төхөөрөмж ашиглаж болно. `--video` нь тайланд байгаа тодорхой гурван дугаартай эх бичлэгийг шалгах зориулалттай. Хуучин алдааны хэрэгслүүдийн зарим тайлангийн хавтас хувилбарын нэртэй хэвээр.

Автомат кодын тестүүдийг `python -m unittest discover -s tests -v` командаар ажиллуулна. GUI, камерын бодит шалгалт цонх/төхөөрөмж ашиглах тул энгийн unit test-ээс тусдаа. Шалгалтын бүртгэлүүд `test_artifacts/` дотор үүснэ.
