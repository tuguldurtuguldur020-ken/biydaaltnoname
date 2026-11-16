# Тохиргооны загварууд

`dataset.example.yaml` нь өөрийн тэмдэглэгээтэй YOLO өгөгдлөөр сургалт хийхэд ашиглах бүтцийн жишээ.

- `path`: өгөгдлийн үндсэн хавтас.
- `train`, `val`, `test`: сургалт, баталгаажуулалт, шалгалтын зургийн зам.
- `names`: class-ийн нэр. Энд нэг class буюу `license_plate` байна.

Өөрийн өгөгдлийн замаар хуулбарлан тохируулж, төслийн үндсэн хавтаснаас:

```powershell
.\.venv\Scripts\python.exe -m tools.train --data datasets/mongolian_plates/data.yaml --epochs 80 --device cpu
```

Програмын өдөр тутмын GUI тохиргоо нь `data/settings.json` дотор хадгалагдана; түүний анхдагч утга `platewatch/settings.py`-ийн `DEFAULTS` хэсэгт бий.
