from __future__ import print_function
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

face_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_upperbody.xml')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    img = cv2.imread(imagePath)
    img = imutils.resize(img, width=min(500, img.shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, minNeighbors=3)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    # show the output images
    cv2.imshow("frame",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
'''import numpy as np
import cv2
img = cv2.imread('G:\DISA\images\p1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
