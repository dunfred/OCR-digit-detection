#!/usr/bin/env python
# coding: utf-8
#author: klenam
# In[ ]:


#Importing libraries
import cv2
import numpy as np
import random as rd


# In[ ]:


def facial_detections(cascade, test_image, scaleFactor = 1.1):
    #Creating a copy of image to prevent changes to original one
    image_copy = test_image.copy()
    
    #Convert test image to gray scale as open cv expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    
    #Apply haar classifier to detect faces
    haar_cascade_face = cv2.CascadeClassifier(cascade)
    faces_rect = haar_cascade_face.detectMultiScale(gray_image, scaleFactor, minNeighbors=5)
    
    #Print out to console the number of faces detected
    #if len(faces_rect) > 0:
        #print("Found %d"%(len(faces_rect)), "feature" if len(faces_rect) == 1 else "features")
    
    r, g, b = rd.randint(0, 256), rd.randint(0, 256), rd.randint(0, 256)
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (r, g, b), 5)
        
    return image_copy

#Defining codec and creating VideoWriter object
class Record:
    def __init__(self, filename):
        self.filename = filename
    
    def rec(self):        
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("Sessions Archieve/%s_session_v%d.avi"%(self.filename, rd.randint(100,901)), fourcc, 3.0, (640, 480))
        return out

class Unique_Cascade:
    def __init__(self, ret, frame, title_text, dim, C, unq_cascade, *args, **kwargs):
        self.ret, self.frame = ret, frame
        self.title_text = title_text
        self.dim = dim
        self.C = C        
        
        self.unq_cascade = unq_cascade                
        self.win = facial_detections(cascade = self.unq_cascade, test_image = self.frame, scaleFactor=1.3)        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = cv2.putText(self.win, self.title_text, self.dim, font, 0.7, self.C, 1, cv2.LINE_AA)        
        return cv2.imshow("Facial Recognition Window", text)


cv2.namedWindow("Facial Recognition Window", 120)

cv2.moveWindow("Facial Recognition Window", 200, 120 )


# In[ ]:


#Start capturing
cap = cv2.VideoCapture(0)

#Initializing haar cascade files
eye_haar_cascade = "data/haarcascade_eye.xml"
face_haar_cascade = "data/haarcascade_frontalface_alt.xml"
left_eye_haar_cascade = "data/haarcascade_lefteye_2splits.xml"
right_eye_haar_cascade = "data/haarcascade_righteye_2splits.xml"
full_body_haar_cascade = "data/haarcascade_fullbody.xml"
upper_body_haar_cascade = "data/haarcascade_upperbody.xml"
#putText parameters initialization
rgb = (255,255,255)
xy = (10, 40)

while True:
    #Capture frame-by-frame
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_PLAIN
    text = cv2.putText(frame, '''**Press and Hold key any key [F, E, L , R, B, U, and quit(Q)] to see what they do.**''',
                       (20, 60), font, 0.7, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow("Facial Recognition Window", text)
    
    if cv2.waitKey(1) & 0xFF == ord("f"):                    
        output_object = Record("Face_Detector").rec()
        while True:                                                                     
            #Display to screen
            text_on_image = 'Detecting Face, Press "r" To Return To Main Screen.'
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, face_haar_cascade)
            #Recording section
            if ret == True:
                output_object.write(obj.win)
            if cv2.waitKey(1) & 0xFF == ord("r"):                
                output_object.release()
                break
                    
    if cv2.waitKey(1) & 0xFF == ord("e"):
        output_object = Record("both_eyes_Detection").rec()
        while True:            
            
            #Display to screen
            text_on_image = 'Detecting both eyes, Press "r" To Return To Main Screen.'
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, eye_haar_cascade)
            #Recording section
            if obj.ret == True:
                output_object.write(obj.win)
            
            if cv2.waitKey(1) & 0xFF == ord("r"):
                output_object.release()
                break
                
    if cv2.waitKey(1) & 0xFF == ord("l"):  
        output_object = Record("left_eye_Detection").rec()
        while True:            
            #Display to screen
            text_on_image = 'Detecting left eye, Press "r" To Return To Main Screen'  
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, left_eye_haar_cascade)
            #Recording section
            if obj.ret == True:
                output_object.write(obj.win)               
            
            if cv2.waitKey(1) & 0xFF == ord("r"):
                output_object.release()
                break
                
    if cv2.waitKey(1) & 0xFF == ord("r"):   
        output_object = Record("right_eye_Detection").rec()
        while True:            
            #Display to screen
            text_on_image = 'Detecting right eye, Press "l" To Return To Main Screen.'  
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, right_eye_haar_cascade)
            #Recording section
            if obj.ret == True:
                output_object.write(obj.win)
                            
            if cv2.waitKey(1) & 0xFF == ord("l"):
                output_object.release()
                break
            
    if cv2.waitKey(1) & 0xFF == ord("b"):
        output_object = Record("full_body_Detection").rec()
        while True:            
            #Display to screen
            text_on_image = 'Detecting full body, Press "r" To Return To Main Screen.' 
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, full_body_haar_cascade)
            #Recording section
            if obj.ret == True:
                output_object.write(obj.win)
                
            if cv2.waitKey(1) & 0xFF == ord("r"):
                output_object.release()
                break
        
    if cv2.waitKey(1) & 0xFF == ord("u"):  
        output_object = Record("upper_body_Detection").rec()
        while True:            
            #Display to screen
            text_on_image ='Detecting upper body, Press "r" To Return To Main Screen.'
            ret, frame = cap.read()
            obj = Unique_Cascade(ret, frame, text_on_image, xy, rgb, upper_body_haar_cascade)
            #Recording section
            if obj.ret == True:
                output_object.write(obj.win)
                
            if cv2.waitKey(1) & 0xFF == ord("r"):
                output_object.release()
                break

                
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
#when everything is done, release capture
cap.release()
#out.release()
cv2.destroyAllWindows()


# In[ ]:




