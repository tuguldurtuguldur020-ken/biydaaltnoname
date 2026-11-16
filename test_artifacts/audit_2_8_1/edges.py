from pathlib import Path
import cv2
import numpy as np

folder=Path(__file__).parent
crop=cv2.imread(str(folder/'detector_crop.png'))
h,w=crop.shape[:2]
big=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
gray=cv2.cvtColor(big,cv2.COLOR_BGR2GRAY)
candidates=[]
for label,binary in [('edge',cv2.Canny(gray,50,150)),('edgeclose',cv2.morphologyEx(cv2.Canny(gray,40,100),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))),('otsu',cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])]:
 cv2.imwrite(str(folder/f'edges_{label}.png'),binary)
 contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 for c in contours:
  area=cv2.contourArea(c)
  if area<gray.size*.08: continue
  poly=cv2.approxPolyDP(c,.03*cv2.arcLength(c,True),True)
  if len(poly)!=4 or not cv2.isContourConvex(poly):continue
  p=poly.reshape(4,2).astype(np.float32)
  sums=p.sum(1);diff=np.diff(p,axis=1).ravel()
  quad=np.array([p[sums.argmin()],p[diff.argmin()],p[sums.argmax()],p[diff.argmax()]],np.float32)
  if len(np.unique(quad,axis=0))!=4:continue
  a=round(max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3])))
  b=round(max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1])))
  if not 1.0<a/b<7:continue
  candidate=cv2.warpPerspective(big,cv2.getPerspectiveTransform(quad,np.float32([(0,0),(a-1,0),(a-1,b-1),(0,b-1)])),(a,b))
  index=len(candidates);cv2.imwrite(str(folder/f'edge_quad_{index}.png'),candidate)
  print(index,label,round(area/gray.size,3),(quad/3).round(1).tolist(),(a,b),flush=True)
  candidates.append(candidate)
canvas=np.full((220*((len(candidates)+3)//4),1000,3),240,np.uint8)
for i,c in enumerate(candidates):
 a,b=c.shape[1],c.shape[0];scale=min(245/a,190/b)
 c=cv2.resize(c,(round(a*scale),round(b*scale)));x,y=(i%4)*250,(i//4)*220
 canvas[y:y+c.shape[0],x:x+c.shape[1]]=c;cv2.putText(canvas,str(i),(x+5,y+210),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
cv2.imwrite(str(folder/'edge_candidates.jpg'),canvas)
