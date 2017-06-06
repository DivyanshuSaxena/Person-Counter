from __future__ import print_function
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

def norepeat(array,tupl):
    wtupl = tupl[1][0] - tupl[0][0]
    htupl = tupl[1][1] - tupl[0][1]
    p = 0.25
    width_na = p*wtupl
    height_na = p*htupl
    bada_box = ((tupl[0][0]-width_na, tupl[0][1]-height_na),(tupl[1][0]+width_na, tupl[1][1]+height_na))
    
    for t in array:
        wt_na = p*(t[1][0] - t[0][0])
        ht_na = p*(t[1][1] - t[0][1])
        bada_t = ((t[0][0]-wt_na, t[0][1]-ht_na),(t[1][0]+wt_na, t[1][1]+ht_na))
        if (t[0][0]>bada_box[0][0] and t[0][1]>bada_box[0][1] and t[1][0]<bada_box[1][0] and t[1][1]<bada_box[1][1]): 
            print("no1")
            return False
        if (tupl[0][0]>bada_t[0][0] and tupl[0][1]>bada_t[0][1] and tupl[1][0]<bada_t[1][0] and tupl[1][1]<bada_t[1][1]):
            print("no2")
            return False
    return True
    

face1_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')
'''face2_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_frontalface_alt.xml')
face3_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_frontalface_alt_tree.xml')
face4_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')'''
prof_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades_cuda\haarcascade_profileface.xml')
head_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_upperbody.xml')
eye_cascade = cv2.CascadeClassifier('G:\DISA\opencv\sources\data\haarcascades\haarcascade_eye.xml')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# loop over the image paths
for imagePath in paths.list_images(args["images"]):
    img = cv2.imread(imagePath)
    print(imagePath)
    arr = []
    img = imutils.resize(img, width=min(800, img.shape[1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces1 = face1_cascade.detectMultiScale(img, 1.05, minNeighbors=3, minSize=(20,20))
    '''faces3 = face3_cascade.detectMultiScale(gray, 1.045, minNeighbors=3, minSize=(20,20))
    faces2 = face2_cascade.detectMultiScale(gray, 1.05, minNeighbors=3, minSize=(20,20))
    faces4 = face4_cascade.detectMultiScale(gray, 1.05, minNeighbors=3, minSize=(20,20))'''
    prof = prof_cascade.detectMultiScale(img, 1.075, minNeighbors=3, minSize=(20,20))
    
    for (x,y,w,h) in faces1:
        if (norepeat(arr,((x,y),(x+w,y+h)))):
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
            arr.append(((x,y),(x+w,y+h)))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            crop = imutils.resize(roi_color, width=400)
            eye = eye_cascade.detectMultiScale(crop, 1.1, minNeighbors=3)
            for (a,b,m,n) in eye:
                cv2.rectangle(crop, (a,b), (a+m,b+n), (0,255,255), 2)
                cv2.imshow("frame",crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    for (x,y,w,h) in prof:
        if norepeat(arr,((x,y),(x+w,y+h))):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            arr.append(((x,y),(x+w,y+h)))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            crop = imutils.resize(roi_color, width=400)
            eye = eye_cascade.detectMultiScale(crop, 1.1, minNeighbors=3)
            for (a,b,m,n) in eye:
                cv2.rectangle(crop, (a,b), (a+m,b+n), (0,255,255), 2)
                cv2.imshow("frame",crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    '''for (x,y,w,h) in head:
        if norepeat(arr,((x,y),(x+w,y+h))):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            arr.append(((x,y),(x+w,y+h)))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

    for (x,y,w,h) in faces4:
        if norepeat(arr,((x,y),(x+w,y+h))):
            cv2.rectangle(img,(x,y),(x+w,y+h),(150,150,150),2)
            arr.append(((x,y),(x+w,y+h)))
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]'''
        
    # show the output images
    cv2.imshow("frame",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
