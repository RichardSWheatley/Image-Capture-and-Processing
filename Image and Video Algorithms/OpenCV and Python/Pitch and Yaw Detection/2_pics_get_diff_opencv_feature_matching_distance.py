# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:07:05 2017

@author: rwheatley
"""

import cv2
import numpy as np
import time
import easygui
from matplotlib import pyplot as plt



def reject_outliers(data, m = 1./2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def runImageProcessing(img_1, img_2, detector, type, startTime, ref_heading, delta_pixels_h, delta_pixels_w):
    height, width = img_1.shape
    crop_img1 = img_1[0:height, delta_pixels_w:width]
    crop_img2 = img_2[0:height, 0:width-delta_pixels_w]

    # find the keypoints and descriptors with "detector"
    (kp1, desc1) = detector.detectAndCompute(crop_img1,None)
    (kp2, desc2) = detector.detectAndCompute(crop_img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.knnMatch(desc1,desc2, k=2);

    good = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
                
    if(len(good) > 0):
        pts_left_image = []
        pts_right_image = []
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.4*n.distance:
                pts_left_image.append(kp1[m.queryIdx].pt)
                pts_right_image.append(kp2[m.trainIdx].pt)
		
        if(len(pts_left_image) > 0):
            pts_left_image = np.float32(pts_left_image)
            pts_right_image = np.float32(pts_right_image)
            
    mine = pts_left_image - pts_right_image
			
    #mine = reject_outliers(mine)
			
    end = time.time()
    print(type, " - Number of matches (total):", len(matches), "Number of matches (good):", len(mine), " - Time: ",  end - startTime)

    if(len(pts_left_image) > 0):
        pts_left_image = np.float32(pts_left_image)
        pts_right_image = np.float32(pts_right_image)
            
        mine = pts_right_image - pts_left_image
        my_heading_string  = "delta-heading: " 
        my_pitch_string    = "delta-pitch:   "
        my_heading_string2 = "ref'd-heading: "
        heading = np.average(mine[:, 0])
        pitch = np.average(mine[:, 1])
        ref_heading = ref_heading + heading*0.05625

    degrees = True
    if degrees == True:
        cv2.putText(img_1, my_pitch_string + "{:10.4f}".format(pitch*0.05625), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);
        cv2.putText(img_1, my_heading_string + "{:10.4f}".format(heading*0.05625), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);
        cv2.putText(img_1, my_heading_string2 + "{:10.4f}".format(ref_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);
    else:
        cv2.putText(img_1, my_pitch_string + "{:10.4f}".format(pitch), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);
        cv2.putText(img_1, my_heading_string + "{:10.4f}".format(heading), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);
        cv2.putText(img_1, my_heading_string2 + "{:10.4f}".format(ref_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA);

# Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(crop_img1,kp1,crop_img2,kp2,good[:len(mine)], None, flags=2)
		
    return(img3, ref_heading)

cv2.ocl.setUseOpenCL(False)
# Initiate SIFT detector
# Initiate SIFT detector
orb = cv2.ORB_create()

reference_heading = 180.0

img1 = cv2.imread(easygui.fileopenbox(), 0)
img2 = cv2.imread(easygui.fileopenbox(), 0)

start = time.time()
done_frame, reference_heading = runImageProcessing(img1.copy(), img2.copy(), cv2.ORB_create(nfeatures=2000), "ORB", start, reference_heading, 0, 267)


cv2.imshow("finished", done_frame)
print(reference_heading)
cv2.imwrite("done_frame_matches.jpg", done_frame)

while(1):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()




