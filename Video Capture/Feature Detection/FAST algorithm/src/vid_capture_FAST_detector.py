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

hyst = hist_type.NORMAL

## Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

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
