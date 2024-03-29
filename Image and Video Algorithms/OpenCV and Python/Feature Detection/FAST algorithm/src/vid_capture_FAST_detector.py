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


import numpy as np
import cv2

cap = cv2.VideoCapture(0)

#hyst = hist_type.NORMAL

## Initiate FAST object with default values
# cv2.FAST_FEATURE_DETECTOR_TYPE_5_8 or
# cv2.FAST_FEATURE_DETECTOR_TYPE_7_12 or
#  cv2.FAST_FEATURE_DETECTOR_TYPE_9_16 or
# Leave blank for Corner Detection
fast = cv2.FastFeatureDetector_create(threshold=5, type=cv2.FAST_FEATURE_DETECTOR_TYPE_5_8)

while(True):
    (isFrame, frame) = cap.read()

    # If not valid frame, break operation
    if not isFrame:
        break


    # find and draw the keypoints
    kp = fast.detect(frame,None)

    img2 = frame.copy()

    #img2 = draw_keypoints(gray, kp, color=(255,0,0))
    cv2.drawKeypoints(frame, kp, img2, color=(255,0,0))

    # show images side by side with and without feature detection
    cv2.imshow('Image', img2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
