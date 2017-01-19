# Copyright (c) 2016 Richard Stephen Wheatley

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture(0)
 
while(True):
    # Capture frame-by-frame
	(isFrame, frame) = cap.read()

	# If not valid frame, break operation
	if not isFrame:
		break

    img2 = frame.copy()

    Red, Green, Blue = cv2.split(frame)

    # Do some denoising on the red channel
    filter_red = cv2.bilateralFilter(Red,25,25,10)

    # Threshold image
    ret, thresh_image = cv2.threshold(filter_red,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find the largest contour and extract it
    im, contours, hierarchy = cv2.findContours(thresh_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour

    # Create a mask from the largest contour
    mask = np.zeros_like(thresh_image)
    cv2.fillPoly(mask,[maxContourData],1)

    # Use mask to crop data from original image
    finalImage = np.zeros_like(frame)
    finalImage[:,:,0] = np.multiply(Red,mask)
    finalImage[:,:,1] = np.multiply(Green,mask)
    finalImage[:,:,2] = np.multiply(Blue,mask)
    cv2.imshow('final',finalImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()