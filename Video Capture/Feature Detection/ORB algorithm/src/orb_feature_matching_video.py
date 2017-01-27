# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:30:41 2017

@author: rwheatley
"""
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture('C://Users//rwheatley//python_anaconda//output.mp4')

while(1):
    ret,frame = cap.read()
    
    # check to see if we have reached the end of the
    # video
    if not ret:
        break

    frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initiate SIFT detector
    orb = cv2.ORB_create()
 
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(frame_gray1,None)

    ret,frame = cap.read()
    
    # check to see if we have reached the end of the
    # video
    if not ret:
        break    
    
    frame_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb.detectAndCompute(frame_gray2,None)
 
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
 
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
        
    # Draw first 10 matches.
    img3 = cv2.drawMatches(frame_gray1,kp1,frame_gray2,kp2,matches,None,flags=2)    
       
    cv2.imshow('img3',img3)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()