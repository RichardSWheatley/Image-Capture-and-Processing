# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:28:31 2017

@author: rwheatley
"""

import numpy as np
import cv2
import easygui
import time

cv2.ocl.setUseOpenCL(False)

def reject_outliers(data, m = 1./2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def runImageProcessing(img_1, img_2, detector, type, startTime):
	# findq the keypoints and descriptors with AKAZE
	(kp1, desc1) = detector.detectAndCompute(img_1,None)
	(kp2, desc2) = detector.detectAndCompute(img_2,None)

	bf = cv2.BFMatcher()
	if(type == "AKAZE" or type == "BRISK"):
		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	
	matches = bf.knnMatch(desc1, desc2, k=2)
	dis_length = len(matches)
	if(len(matches) > 0):
		# my_length_check = 0
		# multiplier = 0.01
		# count = 0
		# while my_length_check < 0.10 and multiplier < 1.0:
			# count += 1 
			# pts_left_image = []
			# pts_right_image = []
			# for i, (m,n) in enumerate(matches):
				# if m.distance < multiplier*n.distance:
					# pts_left_image.append(kp1[m.queryIdx].pt)
					# pts_right_image.append(kp2[m.trainIdx].pt)

			# my_length_check = len(pts_left_image)/dis_length
			# print(my_length_check, multiplier)
			# multiplier += 0.01

		pts_left_image = []
		pts_right_image = []
		for i, (m,n) in enumerate(matches):
			pts_left_image.append(kp1[m.queryIdx].pt)
			pts_right_image.append(kp2[m.trainIdx].pt)
		
		if(len(pts_left_image) > 0):
			pts_left_image = np.float32(pts_left_image)
			pts_right_image = np.float32(pts_right_image)
            
		mine = pts_left_image - pts_right_image
			
		reject_outliers(mine)
			
		end = time.time()
		print(type, " - Number of matches (total):", len(matches), "Number of matches (good):", len(mine), " - Time: ",  end - startTime)

img1 = cv2.imread(easygui.fileopenbox())
img2 = cv2.imread(easygui.fileopenbox())

start = time.time()
runImageProcessing(img1.copy(), img2.copy(), cv2.ORB_create(nfeatures=2000), "ORB", start)

start = time.time()
runImageProcessing(img1.copy(), img2.copy(), cv2.AKAZE_create(), "AKAZE", start)

start = time.time()
runImageProcessing(img1.copy(), img2.copy(), cv2.BRISK_create(thresh=160, octaves=1), "BRISK", start)

