# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:07:05 2017

@author: rwheatley
"""

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)
# Initiate AKAZE detector
detector = cv2.AKAZE_create(nfeatures=1000)

stream = cv2.VideoCapture(0)

# grab first image
ret, new_frame = stream.read()
new_frame = cv2.flip(new_frame, 1)
old_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',new_frame)
cv2.waitKey(1)
reference_heading = 0.00
offset = 0.00

while(stream.isOpened()):
    ret, new_frame = stream.read()

    if ret:
        new_frame = cv2.flip(new_frame, 1)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # findq the keypoints and descriptors with AKAZE
        (kp1, desc1) = detector.detectAndCompute(old_gray,None)
        (kp2, desc2) = detector.detectAndCompute(new_gray,None)

        # Use Hamming distance, because AKAZE uses binary descriptor by default.
		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        if(len(matches) > 0):
            my_length = 0
            multiplier = 0.001
            count = 0
            while my_length < 10 and multiplier < 1.0:
                count += 1 
                pts_left_image = []
                pts_right_image = []
                for i, (m,n) in enumerate(matches):
                    if m.distance < multiplier*n.distance:
                        pts_left_image.append(kp1[m.queryIdx].pt)
                        pts_right_image.append(kp2[m.trainIdx].pt)
                    my_length = len(pts_left_image)
                    multiplier += 0.00025
      
            print("Count: ", count, "Number of matches (total):", len(matches), "Number of matches (good):", len(pts_left_image))


            if(len(pts_left_image) > 0):
                pts_left_image = np.float32(pts_left_image)
                pts_right_image = np.float32(pts_right_image)
            
                mine = pts_right_image - pts_left_image
                my_heading_string  = "delta-heading: " 
                my_pitch_string    = "delta-pitch:   "
                heading = np.average(mine[:, 0])
                pitch = np.average(mine[:, 1])
                reference_heading = heading + offset
				
                degrees = 0.05625
                cv2.putText(new_frame, my_pitch_string + "{:10.4f}".format(pitch*degrees), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                cv2.putText(new_frame, my_heading_string + "{:10.4f}".format(reference_heading*degrees), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);

        cv2.imshow('frame',new_frame)      
    key_check = cv2.waitKey(1) & 0xFF
    if key_check == ord('q'):
        break
    elif key_check == ord('s'):
        old_gray = new_gray.copy()
        offset = reference_heading

stream.release()
cv2.destroyAllWindows()




