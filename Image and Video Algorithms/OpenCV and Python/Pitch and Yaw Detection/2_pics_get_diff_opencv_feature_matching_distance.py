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

def reject_outliers(data, m=2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def find_good_matches(data):
    good = []
    for m,n in data:
        if m.distance < 0.8*n.distance:
            good.append([m])
    return good

def find_better_matches(data):
        pts_left_image = []
        pts_right_image = []

        for i, (m,n) in enumerate(data):
        if m.distance < 0.33*n.distance:
            pts_left_image.append(kp1[m.queryIdx].pt)
            pts_right_image.append(kp2[m.trainIdx].pt)
    return(pts_left_image, pts_right_image)

def runImageProcessing(img_1, img_2, detector, type, startTime, ref_heading, delta_pixels_h, delta_pixels_w):
    height, width = img_1.shape
    print(height, width)

    crop_img1 = img_1[0:height, 0:width]
    crop_img2 = img_2[0:height, 0:width]

    if delta_pixels_h > 0:
        if delta_pixels_w > 0:
            crop_img1 = img_1[delta_pixels_h:height, delta_pixels_w:width]
            crop_img2 = img_2[0:height-delta_pixels_h, 0:width-delta_pixels_w]
        elif delta_pixels_w < 0:
            crop_img1 = img_1[delta_pixels_h:height, 0:width-delta_pixels_w]
            crop_img2 = img_2[0:height-delta_pixels_h, delta_pixels_w:width]
    else:
        if delta_pixels_w > 0:
            crop_img1 = img_1[0:height-delta_pixels_h, delta_pixels_w:width]
            crop_img2 = img_2[delta_pixels_h:height, 0:width-delta_pixels_w]
        elif delta_pixels_w < 0:
            crop_img1 = img_1[0:height-delta_pixels_h, 0:width-delta_pixels_w]
            crop_img2 = img_2[delta_pixels_h:height, delta_pixels_w:width]    

    # find the keypoints and descriptors with "detector"
    (kp1, desc1) = detector.detectAndCompute(crop_img1,None)
    (kp2, desc2) = detector.detectAndCompute(crop_img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors.
    matches = bf.knnMatch(desc1,desc2, k=2)

    good = find_good_matches(matches)

    if(len(good) > 0):
        pts_left_image, pts_right_image = find_better_matches(good)

        if(len(pts_left_image) > 0):
            pts_left_image = np.float32(pts_left_image)
            pts_right_image = np.float32(pts_right_image)
            
            mine = pts_left_image - pts_right_image

            mine = reject_outliers(mine)

            end = time.time()
            print(type, " - Number of matches (total):", len(matches), "Number of matches (good):", len(mine), " - Time: ",  end - startTime)

            my_heading_string  = "delta-heading: " 
            my_pitch_string    = "delta-pitch:   "
            my_heading_string2 = "ref'd-heading: "
            heading = np.average(mine[:, 0])
            pitch = np.average(mine[:, 1])
            ref_heading = ref_heading + heading*0.075

            # If we want different units
            units = "pixels"

            # Draw matches that we accept
            img3 = cv2.drawMatchesKnn(crop_img1,kp1,crop_img2,kp2,good[:len(mine)], None, flags=2)

            if units == "degrees":
                cv2.putText(img3, my_pitch_string + "{:10.4f}".format(pitch*0.075), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
                cv2.putText(img3, my_heading_string + "{:10.4f}".format(heading*0.075), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
                cv2.putText(img3, my_heading_string2 + "{:10.4f}".format(ref_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
            elif units == "mils":
                cv2.putText(img3, my_pitch_string + "{:10.4f}".format(pitch), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
                cv2.putText(img3, my_heading_string + "{:10.4f}".format(heading), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
                cv2.putText(img3, my_heading_string2 + "{:10.4f}".format(ref_heading), (30,90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
            else # This is for Pixel Measurements
                cv2.putText(img3, "delta-pixels-pitch: " + "{:10.4f}".format(pitch), (30,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)
                cv2.putText(img3, "delta-pixels-yaw: " + "{:10.4f}".format(heading), (30,60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,250), 1, cv2.LINE_AA)

            return(img3, ref_heading)
    return(img_2, 999.99)

def main():
    cv2.ocl.setUseOpenCL(False)

    # Initiate ORB detector
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

if __name__ == '__main__':
    main()