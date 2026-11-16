import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from platewatch.settings import load_settings
from platewatch.recognition import PlateRecognizer, crop_variants
from platewatch.cyrillic_ocr import light_plate
from platewatch.geometry import plate_quadrilaterals
from platewatch.media import read_image
from platewatch.ocr import text_regions

folder=Path(__file__).parent
source=Path.home()/'OneDrive/Pictures/846220724_1123758723635137_4283606675668681012_n.jpg'
frame=read_image(source)
r=PlateRecognizer(load_settings())
boxes=r.detect_detailed(frame)
print('boxes',boxes,flush=True)
x1,y1,x2,y2=boxes[0].bbox
crop=frame[y1:y2,x1:x2]
cv2.imwrite(str(folder/'detector_crop.png'),crop)
cv2.imwrite(str(folder/'light_crop.png'),light_plate(crop))
print('shapes',crop.shape,light_plate(crop).shape,flush=True)
for mode in (False,True):
 print('primary',mode,r.cyrillic_reader.read_plate(crop,detailed=mode),flush=True)
 print('legacy',mode,r._read_crop_text(crop,detailed=mode),flush=True)
for label,image in [('frame',frame),('crop',crop)]:
 proposals=plate_quadrilaterals(image)
 print('proposals',label,len(proposals),flush=True)
 for i,(box,quad) in enumerate(proposals):
  cv2.imwrite(str(folder/f'quad_{label}_{i}.png'),quad)
  print('quadread',box,r.cyrillic_reader.read_plate(quad,detailed=True),flush=True)
for i,points in enumerate(([(110,416),(187,433),(176,500),(98,480)],[(115,422),(175,436),(164,485),(106,471)])):
 src=np.float32(points);dst=np.float32([(0,0),(159,0),(159,127),(0,127)])
 image=cv2.warpPerspective(frame,cv2.getPerspectiveTransform(src,dst),(160,128))
 cv2.imwrite(str(folder/f'manual_rect_{i}.png'),image)
 print('manual rect',i,r.cyrillic_reader.read_plate(image,detailed=True),r._read_crop_text(image,detailed=True),flush=True)
for idx,image in enumerate(crop_variants(crop,detailed=True)):
 regions=text_regions(image)
 print('regions',idx,regions,flush=True)
 cv2.imwrite(str(folder/f'variant_{idx}.png'),image)
