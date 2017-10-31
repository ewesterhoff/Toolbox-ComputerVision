""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

def draw_eye(img, center):
    cv2.circle(img, center, int(20), (int(255),int(255),int(255)), -1)
    cv2.circle(img, center, int(7), (int(255),int(0),int(0)), -1)
    cv2.circle(img, center, int(3), (int(0),int(0),int(0)), -1)

def draw_face(x,y,w,h,img):
    center_eye1 = (int(x+(.3*h)), int(y+(.34*w)))
    center_eye2 = (int(x+(.7*h)), int(y+(.34*w)))
    center_sad = (int(x+(.5*h)), int(y+(.8*w)))
    draw_eye(img, center_eye1)
    draw_eye(img, center_eye2)
    cv2.ellipse(img, center_sad, (int(40), int(20)), 0, 190, 350, (int(0),int(0),int(0)), thickness=4)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/Users/ewesterhoff/Downloads/haarcascade_frontalface_alt.xml')
kernel = np.ones((50,50), 'uint8')


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        draw_face(x,y,w,h, frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
