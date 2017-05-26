# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:30:41 2017

@author: rwheatley
"""
import numpy as np
import cv2
from Tkinter import Tk
from tkFileDialog import askopenfilename

cv2.ocl.setUseOpenCL(False)

def get_filename():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return filename

def orb_find_features():

   filename = get_filename()
   print(filename)
   
   if filename:
      cap = cv2.VideoCapture(filename)
   
      while(1):
          ret,frame = cap.read()
          
          # check to see if we have reached the end of the
          # video
          if not ret:
              break
      
          frame_gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
          # Initiate ORB detector
          # default values are:
          # int nfeatures=500
          # float scaleFactor=1.2f
          # int nlevels=8
          # int edgeThreshold=31
          # int firstLevel=0
          # int WTA_K=2
          # int scoreType=ORB::HARRIS_SCORE
          # int patchSize=31
          orb = cv2.ORB_create()
       
          # find the keypoints and descriptors with ORB
          kp, des = orb.detectAndCompute(frame_gray1,None)
      
          img2 = cv2.drawKeypoints(frame_gray1,kp,None,color=(0,255,0), flags=0) 
             
          cv2.imshow('img2',img2)
      
          k = cv2.waitKey(30) & 0xff
          if k == 27:
              break
   
   
   cv2.destroyAllWindows()
   cap.release()
   
##        
def main():
    orb_find_features()
    ##print_faces()
    ##wait_for_user()
    
if __name__ == "__main__":
    main()