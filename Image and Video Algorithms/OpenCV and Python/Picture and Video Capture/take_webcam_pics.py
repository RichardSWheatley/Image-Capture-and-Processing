# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:30:41 2017

@author: rwheatley
"""
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)

def orb_find_features():
   """Find features in an image using the ORB detector"""
   print(cv2.__version__)
   print(np.__version__)

   cap = cv2.VideoCapture(0)
   count = 0

   while(1):
       ret,frame = cap.read()
       
       # check to see if we have reached the end of the video
       if not ret:
           break
       
       cv2.imshow("frame",frame)
   
       k = cv2.waitKey(1) & 0xFF
 
       if k == ord('q'): # wait for ESC key to exit
           break
       elif k == ord('s'): # wait for 's' key to save and exit
           cv2.imwrite('img_rects' + str(count) + '.jpg',frame)   
           count += 1
           
   cv2.destroyAllWindows()
   cap.release()     
def main():
   """My main mayun"""
    orb_find_features()
    ##print_faces()
    ##wait_for_user()
    
if __name__ == "__main__":
    main()