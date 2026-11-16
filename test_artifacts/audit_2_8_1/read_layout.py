from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
from platewatch.recognition import PlateRecognizer
from platewatch.layout import two_row_crops
from platewatch.settings import load_settings
from platewatch.plates import ALLOWLIST
folder=Path(__file__).parent
r=PlateRecognizer(load_settings())
for name in ('detector_crop','deskew_13.7','manual_rect_0'):
 image=cv2.imread(str(folder/(name+'.png')))
 crops=list(two_row_crops(image))
 print(name,'pairs',len(crops),flush=True)
 for i,pair in enumerate(crops):
  for j,row in enumerate(pair):
   cv2.imwrite(str(folder/f'row_{name}_{i}_{j}.png'),row)
   print(j,'pp',r.cyrillic_reader.read_line(row),flush=True)
   gray=cv2.cvtColor(row,cv2.COLOR_BGR2GRAY)
   print(j,'easy',r.reader.recognize(gray,horizontal_list=[[0,gray.shape[1],0,gray.shape[0]]],free_list=[],detail=1,paragraph=False,allowlist='0123456789' if j==0 else ALLOWLIST[10:],workers=0,batch_size=1),flush=True)
