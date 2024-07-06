#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os 
import cv2 as cv 
import ultralytics
from sort import *


# In[27]:


from ultralytics import YOLO


# In[72]:


# Use 'best_7.pt' to count total number of mask came in the video. Here mask class number is 0 need to adjust code
# Use '2_class.pt' to detect face with and without mask. mask = 1, mask =2
print('''There are 3 functions here i.e. mask_detection(),  video() and save_video(). All will  detect face with mask and without
mask wtih confidence level. They will also count total number of face with masks in the video and for the single frame''')
print('''As the name suggests, video will just play video with detection whereas save_video will save the video as well. In mask_detection(), it will detect
face with mask or not in the image''')


# In[28]:


pred= YOLO('2_class.pt')


# In[70]:


def mask_detector(inp):
    img = cv.imread(inp)
    result = pred(img, iou = 0.1)
    mc = 0
    nc = 0
    for r in result:
        cl_name = r.names
        for box in r.boxes.data:
            
            x1, y1, x2, y2, conf, clas = box
            x1, y1, x2, y2, conf, clas = int(x1), int(y1), int(x2), int(y2), int(conf*100), int(clas)
            
            text = cl_name[clas] +f': {conf}'
            cv.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
            if clas == 1:
                mc +=1
                cv.putText(img, text, (max(0,x1-5), max(10, y1-5)), 
                      cv.FONT_HERSHEY_COMPLEX, 0.5, (25, 255 ,105),1)
            else:
                nc +=1
                cv.putText(img, text, (max(0,x1-5), max(10, y1-5)), 
                      cv.FONT_HERSHEY_COMPLEX, 0.5, (25, 255 ,105),1)
    cv.putText(img, f'Face with Masks: {mc}' ,(10,20),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),1)
    cv.putText(img, f'Face with no Mask: {nc}' ,(10,50),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),1 )
    
    cv.imshow('Result', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
                


# In[ ]:





# In[22]:


def save_video():
    inp = input('Give path to the video')
    out = input('Give path for the output')
    

    
    
    cap = cv.VideoCapture(inp)
    framewidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameheight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    output = cv.VideoWriter(out, fourcc, fps, (framewidth, frameheight))
    tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)
    #limit = [0,200,620,200]
    
    track_id = []
    while True:
        ret, img = cap.read()
        if ret == True:
            results = pred(img, stream = True, iou = 0.1, conf = 0.35)
            detection = np.empty((0,5))
            count = 0
            for result in results:
                
                for i in result.boxes:
                    
                    cls = i.cls
                    if int(cls) == 1:
                        count +=1
                        conf = i.conf
                        x1,y1,x2,y2 = i.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        #cv.rectangle(img, (x,y), (x1,y1), (255,0,255),2)
                        #cv.putText(img, f'Mask{round(float(conf),2)}' ,(x,y-10),cv.FONT_HERSHEY_COMPLEX, 0.8, (25, 155 ,125),2)
                        array = np.array([x1,y1,x2,y2, int(conf*100)])
                        detection = np.vstack((detection, array))
            
                    elif int(cls) == 0:
                        conf = i.conf
                        x1,y1,x2,y2 = i.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        cv.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
                        cv.putText(img, f'FACE{round(float(conf),2)}' ,(x1,y1-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (25, 155 ,125),2)
                        
            track = tracker.update(detection)
           
            for tr in track:
                x1,y1,x2,y2,iden = tr
                x1,y1,x2,y2,iden = int(x1),int(y1),int(x2),int(y2), int(iden)
                cv.rectangle(img, (x1,y1),(x2,y2), (255,0,255), 2)
                cv.putText(img, f'Mask ID: {str(iden)} ', (max(0,x1-15), max(35, y1-5)), 
                              cv.FONT_HERSHEY_COMPLEX, 0.5, (50,125, 0), 2)
                #cx, cy = x1+ (x2-x1)//2, y1 +(y2-y1)
               
           
                if (iden not in track_id):
                #if( limit[0]<cx<limit[2]) &( limit[1] -20<cy< limit[1] + 2) &(iden not in track_id):
                    
                    track_id.append(iden)
            cv.putText(img, f'Total Masks: {len(track_id)}' ,(10,20),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),2 )
            cv.putText(img, f'Per scene: {count}' ,(10,50),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),2 )
                
            
            cv.imshow('img', img)
            output.write(img)
            if cv.waitKey(1) & 0xFF == ord('x'):
                break
        else:
            break
            
    output.release()
    cap.release()
    cv.destroyAllWindows()
    


# In[23]:


def video():
    
    path = input('Give path of the video')
    cap = cv.VideoCapture(path)
    tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)


    track_id = []
    while True:
        ret, img = cap.read()
        if ret == True:
            results = pred(img, stream = True, iou = 0.1, conf = 0.35)
            detection = np.empty((0,5))
            count = 0
            for result in results:
                
                for i in result.boxes:
                    
                    cls = i.cls
                    if int(cls) == 1:
                        count +=1
                        conf = i.conf
                        x1,y1,x2,y2 = i.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        #cv.rectangle(img, (x,y), (x1,y1), (255,0,255),2)
                        #cv.putText(img, f'Mask{round(float(conf),2)}' ,(x,y-10),cv.FONT_HERSHEY_COMPLEX, 0.8, (25, 155 ,125),2)
                        array = np.array([x1,y1,x2,y2, int(conf*100)])
                        detection = np.vstack((detection, array))
            
                    elif int(cls) == 0:
                        conf = i.conf
                        x1,y1,x2,y2 = i.xyxy[0]
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        cv.rectangle(img, (x1,y1), (x2,y2), (255,0,255),2)
                        cv.putText(img, f'FACE{round(float(conf),2)}' ,(x1,y1-10),cv.FONT_HERSHEY_COMPLEX, 0.8, (25, 155 ,125),2)
                        
            track = tracker.update(detection)
           
            for tr in track:
                x1,y1,x2,y2,iden = tr
                x1,y1,x2,y2,iden = int(x1),int(y1),int(x2),int(y2), int(iden)
                cv.rectangle(img, (x1,y1),(x2,y2), (255,0,255), 2)
                cv.putText(img, f'Mask ID: {str(iden)} ', (max(0,x1-15), max(35, y1-5)), 
                              cv.FONT_HERSHEY_COMPLEX, 0.5, (50,125, 0), 2)
                #cx, cy = x1+ (x2-x1)//2, y1 +(y2-y1)
               
           
                if (iden not in track_id):
                #if( limit[0]<cx<limit[2]) &( limit[1] -20<cy< limit[1] + 2) &(iden not in track_id):
                    
                    track_id.append(iden)
            cv.putText(img, f'Total Masks: {len(track_id)}' ,(10,20),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),2 )
            cv.putText(img, f'Per scene: {count}' ,(10,50),cv.FONT_HERSHEY_COMPLEX, 0.6, (100, 0 ,255),2 )
                
            
            cv.imshow('img', img)
           
            if cv.waitKey(1) & 0xFF == ord('x'):
                break
        else:
            break
            
 
    cap.release()
    cv.destroyAllWindows()

    


# In[25]:


#Press x to close
print('''Choose the option:
press 1 to play the video only
press 2 to play and save the video as well
press 3 to detect mask on face in images
press x to exit''')
while True:
    x = input('Your input')
    if x == '1':
        video()
        print('''Choose the option:
        press 1 to play the video only
        press 2 to play and save the video as well
        press 3 to detect mask on face in images
        press x to quit''')
        x = input('Your input')
    elif x == '2':
        save_video()
        print('''Choose the option:
        press 1 to play the video only
        press 2 to play and save the video as well
        press 3 to detect mask on face in images
        press x to quit''')
        x = input('Your input')
    elif x == '3':
        mask_detection()
        print('''Choose the option:
        press 1 to play the video only
        press 2 to play and save the video as well
        press 3 to detect mask on face in images
        press x to quit''')
        x = input('Your input')
    elif x not in ['x', '1', '2', '3']:
        print('try again')
        x = input('Your input')
       
    else:
         break
    break
        
        


# In[ ]:




