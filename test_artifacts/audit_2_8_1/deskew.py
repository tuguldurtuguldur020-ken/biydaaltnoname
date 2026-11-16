from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2,numpy as np
from platewatch.recognition import PlateRecognizer
from platewatch.settings import load_settings
from platewatch.media import read_image
from platewatch.plates import select_ocr_candidate,ALLOWLIST
folder=Path(__file__).parent
r=PlateRecognizer(load_settings())
frame=read_image(Path.home()/'OneDrive/Pictures/846220724_1123758723635137_4283606675668681012_n.jpg')
print('960',r._detect_region(frame,960,detailed=True),flush=True)
crop=cv2.imread(str(folder/'detector_crop.png'))
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
lines=cv2.HoughLinesP(cv2.Canny(gray,40,100),1,np.pi/180,20,minLineLength=crop.shape[1]*.3,maxLineGap=10)
angles=[]
for line in lines.reshape(-1,4):
 x1,y1,x2,y2=line; angle=np.degrees(np.arctan2(y2-y1,x2-x1))
 if abs(angle)<=35:angles.append(float(angle))
print('angles',angles,flush=True)
angle=float(np.median(angles))
h,w=crop.shape[:2]
for angle in (angle,10,15):
 matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
 image=cv2.warpAffine(crop,matrix,(w,h),borderMode=cv2.BORDER_REPLICATE)
 cv2.imwrite(str(folder/f'deskew_{angle:.1f}.png'),image)
 print('rot',angle,r.cyrillic_reader.read_plate(image,detailed=True),flush=True)
 # Diagnostic crop trims are deliberately not used in production.
 for left,top,right,bottom in ((60,25,143,93),(67,25,140,93),(65,29,136,90)):
  part=image[top:bottom,left:right]
  print('trim',angle,(left,top,right,bottom),r.cyrillic_reader.read_plate(part,detailed=True),flush=True)
  cv2.imwrite(str(folder/f'trim_{angle:.1f}_{left}_{top}.png'),part)
for i in (0,1):
 image=cv2.imread(str(folder/f'manual_rect_{i}.png'))
 for label,view in [('src',image),('gray',cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)),('contrast',cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)).apply(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)))]:
  print('manual lines',i,label,'all',r.cyrillic_reader.read_line(view),flush=True)
  for split in (.5,.55,.6,.65):
   cut=round(view.shape[0]*split)
   print(split,r.cyrillic_reader.read_line(view[:cut]),r.cyrillic_reader.read_line(view[cut:]),flush=True)
