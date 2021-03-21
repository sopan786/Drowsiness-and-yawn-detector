#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import dlib
import imutils 
from imutils import face_utils 
from scipy.spatial import distance
from playsound import playsound


# In[3]:


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


# In[4]:


def calculate_MAR(mouth): 
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    MAR = (A + B + C) / 3.0
    return MAR


# In[ ]:


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
MAR_THRESHOLD = 14
eye_blink = 0
frame_continue = 60
sleep = 0


# In[5]:


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        mouth = []
        
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
            
        for n in range(48,68):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            mouth.append((x,y))
            next_point = n+1
            if n == 67:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR<0.18:
            eye_blink += 1
            if eye_blink >= frame_continue:
                sleep += 1
                cv2.imwrite("dataset/sleep%d.jpg" % sleep, frame)
                cv2.putText(frame,"DROWSY",(20,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
                playsound('sound files/sleep.mp3')
                print("dontsleep")
            print(EAR)
        
        mouth_mar = calculate_MAR(mouth)
        if mouth_mar > MAR_THRESHOLD:
            playsound('sound files/yawn.mp3')
            cv2.putText(frame, "you are yawning", (10, 400), cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255),4)
            
       
    cv2.imshow("how you are feeling", frame)

    key = cv2.waitKey(1)
    
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




