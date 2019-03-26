import cv2
import numpy as np

face_cas=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        #cv2.rectangle(<where>,...)
        print(x,y,w,h)
        cv2.rectangle(frame, (x+w//4,y+h), (x+w-w//4,y+h+h//4), (255,255,255), 1)
        #(src, lower_left coordinate,upper_right coordinate, color, border_width)
        #cv2.rectangle(frame, (x,y
    cv2.imshow('img',frame)
    
    k=cv2.waitKey(30) & 0xff
    if k==27 or 0xff == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
