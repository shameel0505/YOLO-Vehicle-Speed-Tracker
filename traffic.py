from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections as cl
import time
video='sample.mp4'
vd=cv2.VideoCapture(video)
model=YOLO('yolo11n.pt')
up={}
down={}

xcut=614
yrt=550
speeddown=450
ylt=350
speedup=350
dist=18
def speedest(id,cy,cx):
    if id not in up and cx<xcut and cy>speedup:
        up[id]=time.time()
    elif id not in down and cx<xcut and cy>speeddown:
        down[id]=time.time()
    if id not in up and cx>xcut and cy<speeddown:
        up[id]=time.time()
    elif id not in down and cx>xcut and cy<speedup:
        down[id]=time.time()
    delta=abs(up.get(id,0)-down.get(id,0))
    if id in up and down:
        return (dist/delta)*3.6
    else:
        return 0   
def avgspeed(speed):
    actvalues=list( x for x in speed if x is not None and 30<x<100 )
    if not actvalues:
        return None
    return np.mean(actvalues)
right=cl.defaultdict(int)
left=cl.defaultdict(int)
crossed=set()
ret=True
while ret:
    ret,frm=vd.read()
    results=model.track(frm,persist=True,classes=[2,3,7],conf=.6)
    # print(results)
    class_list=model.names
    # print(class_list)
    # frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    speed={}
    # plt.imshow(frm_rgb)
    # plt.title("Frame Preview")
    # plt.axis('off')  # Hide axis ticks
    # plt.show()
    # cv2.line(frm,(xcut,300),(1410,300),(0,0,255),2)
    cv2.line(frm,(xcut,0),(xcut,720),(0,0,255),2)
    if results[0].boxes.data is not None:
        boxes=results[0].boxes.xyxy
        ids=results[0].boxes.id.int().tolist()
        class_ids=results[0].boxes.cls.int().tolist()
        confidences=results[0].boxes.conf
    vehicles=set()
    for box,id,class_id,confidence in zip(boxes,ids,class_ids,confidences):
        class_name=class_list[class_id]
        vehicles.add(id)
        x1,y1,x2,y2=map(int,box)
        cx=(x1+x2)//2
        cy=(y1+y2)//2
        speed[id]=(speedest(id,cy,cx))
        # cv2.circle(frm,(cx,cy),1,(0,255,0),1,-1)
        # print(speed)
        # average_speed=avgspeed(speed)
         
        cv2.putText(frm,f'ID {id} {class_name} speed={speed[id]}',(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,122,0),2)
        cv2.rectangle(frm,(x1,y1),(x2,y2),(122,255,0),2)
        if cx<xcut and cy>ylt and cy<450 and id not in crossed:
            crossed.add(id)
            left[class_name]+=1
        if cy<yrt and cx>xcut and cy>300 and id not in crossed:
            crossed.add(id)
            right[class_name]+=1
        cv2.putText(frm,f'LEFT LANE',(50,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        y_offset=60
        for vehicle in left:
            cv2.putText(frm,f'{vehicle}: {left[vehicle]}',(50,y_offset),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            y_offset+=30
        cv2.putText(frm,f'RIGHT LANE',(900,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        y_offset=60
        for vehicle in right:
            cv2.putText(frm,f'{vehicle}: {right[vehicle]}',(900,y_offset),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            y_offset+=30
    cv2.putText(frm,f'TRAFFIC STATUS',(50,600),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    if len(vehicles)>20:
        cv2.putText(frm,f'high traffic',(50,630),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    elif len(vehicles)>7 and len(vehicles)<20:
        cv2.putText(frm,f'moderate traffic',(50,630),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
    else:
        cv2.putText(frm,f'low traffic',(50,630),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    # cv2.putText(frm,f'AVERAGE SPEED: {average_speed} km/hr',(50,680),cv2.FONT_HERSHEY_COMPLEX,1,(0,230,0),2)
    # cv2.line(frm,(0,speeddown),(1500,speeddown),(0,0,230),2)
    # cv2.putText(frm,'speeddown',(0,speeddown),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,230),2)
    # cv2.line(frm,(0,speedup),(1500,speedup),(0,0,230),2)
    # cv2.putText(frm,'speedup',(0,speedup),cv2.FONT_HERSHEY_COMPLEX,.2,(0,120,230),1)
    cv2.line(frm,(730,yrt),(1300,yrt),(0,0,230),2)
    cv2.line(frm,(55,ylt),(560,ylt),(0,0,230),2)
    # cv2.imwrite('forcalc.jpg',frm)
    # print(speedline)
        


    cv2.imshow('',frm)
    if cv2.waitKey(1)& 0xff==ord('q'):
        break
vd.release()
cv2.destroyAllWindows()