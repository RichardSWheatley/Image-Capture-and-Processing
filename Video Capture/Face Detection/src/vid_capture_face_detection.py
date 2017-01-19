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
from enum import Enum

class hist_type(Enum):
    NORMAL = 1
    CLAHE = 2

face_cascade_front = cv2.CascadeClassifier('C:\\Python27\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml')
face_cascade_profile = cv2.CascadeClassifier('C:\\Python27\\opencv\\build\\etc\\haarcascades\\haarcascade_profileface.xml')
##face_cascade_front = cv2.CascadeClassifier('C:\\Python27\\opencv\\build\\etc\\lbpcascades\\lbpcascade_frontalface.xml')
##face_cascade_profile = cv2.CascadeClassifier('C:\\Python27\\opencv\\build\\etc\\lbpcascades\\lbpcascade_profileface.xml')
##eye_cascade = cv2.CascadeClassifier('C:\\Python27\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

hyst = hist_type.NORMAL

while(True):
    # Capture frame-by-frame
	(isFrame, frame) = cap.read()

	# If not valid frame, break operation
	if not isFrame:
		break

    # clone frame for side by side comparision
    img2 = frame.copy()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# pick which type of histogram to use
    if hyst == hist_type.NORMAL:
        gray = cv2.equalizeHist(gray)
    elif hyst == hist_type.CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

	# outline faces if they exist
    faces = face_cascade_front.detectMultiScale(gray, 1.1, 3)
    if len(faces) == 0:
        faces = face_cascade_profile.detectMultiScale(gray, 1.1, 3)
    for (x,y,w,h) in faces:
        cv2.ellipse(img2, (x + w/2, y + h/2), (2*w/5, 3*h/5), 0, 0, 360, 255, 2, 4, 0)

    # show images side by side with and without face detection
	res = np.hstack((frame,img2))

    # Display the resulting frame
    cv2.imshow('res',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
