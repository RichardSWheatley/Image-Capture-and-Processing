# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:07:05 2017

@author: rwheatley
"""

import cv2
import numpy as np
cv2.ocl.setUseOpenCL(False)
# Initiate SIFT detector
orb = cv2.ORB_create(nfeatures=1000)

stream = cv2.VideoCapture(0)
ret, new_frame = stream.read()

# grab first image
new_frame = cv2.flip(new_frame, 1)
new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',new_frame)
cv2.waitKey(1)

reference_heading = 180.0

while(stream.isOpened()):
    old_gray = new_gray.copy()
    ret, new_frame = stream.read()

    if ret:
        new_frame = cv2.flip(new_frame, 1)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # findq the keypoints and descriptors with ORB
        (kp1, desc1) = orb.detectAndCompute(old_gray,None)
        (kp2, desc2) = orb.detectAndCompute(new_gray,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        if(len(matches) > 0):
            my_length = 0
            multiplier = 0.001
            count = 0
            while my_length < 1 and multiplier < 1.0:
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
                my_heading_string2 = "ref'd-heading: "
                heading = np.average(mine[:, 0])
                pitch = np.average(mine[:, 1])
                reference_heading = reference_heading + heading*0.05625

                degrees = True
                if degrees == True:
                    cv2.putText(new_frame, my_pitch_string + "{:10.4f}".format(pitch*0.05625), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                    cv2.putText(new_frame, my_heading_string + "{:10.4f}".format(heading*0.05625), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                    cv2.putText(new_frame, my_heading_string2 + "{:10.4f}".format(reference_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                else:
                    cv2.putText(new_frame, my_pitch_string + "{:10.4f}".format(pitch), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                    cv2.putText(new_frame, my_heading_string + "{:10.4f}".format(heading), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);
                    cv2.putText(new_frame, my_heading_string2 + "{:10.4f}".format(reference_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200,200,250), 1, cv2.LINE_AA);

        cv2.imshow('frame',new_frame)      
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()




