# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:20:42 2019

@author: kaustav
"""
import cv2
import numpy as np

eye_cas=cv2.CascadeClassifier('haarcascade_eye.xml')
face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detect_MultiScale(grey)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)  #(src, lower_left coordinate,upper_right coordinate, color, border_width)
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eye_cas.detectMultiScale(roi_grey)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,0,255), 1)
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
capture.release()
cv2.destroyAllWindows()