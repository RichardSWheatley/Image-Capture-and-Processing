# USAGE
# python track.py --video video/sample.mov

# import the necessary packages
# import the necessary packages
from collections import deque
import numpy as np
import argparse
import cv2

# initialize the current frame of the video, along with the list of
# ROI points along with whether or not this is input mode
frame = None
roiPts = []
inputMode = False

pts = deque(maxlen=64)

def selectROI(event, x, y, flags, param):
	# grab the reference to the current frame, list of ROI
	# points and whether or not it is ROI selection mode
	global frame, roiPts, inputMode

	# if we are in ROI selection mode, the mouse was clicked,
	# and we do not already have four points, then update the
	# list of ROI points with the (x, y) location of the click
	# and draw the circle
	if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
		roiPts.append((x, y))
		cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("frame", frame)

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help = "path to the (optional) video file")
	args = vars(ap.parse_args())

	# grab the reference to the current frame, list of ROI
	# points and whether or not it is ROI selection mode
	global frame, roiPts, inputMode, roiHist

	# if the video path was not supplied, grab the reference to the
	# camera
	if not args.get("video", False):
		camera = cv2.VideoCapture(0)

	# otherwise, load the video
	else:
		camera = cv2.VideoCapture(args["video"])

	# setup the mouse callback
	cv2.namedWindow("frame")
	cv2.setMouseCallback("frame", selectROI)

	# keep looping over the frames
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()

		if not grabbed:
			break

		gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		(grabbed, frame) = camera.read()

		if not grabbed:
			break

		gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		diff = cv2.absdiff(gray1, gray2)
        
		ret, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
		blurred = cv2.blur(thresh,(10,10))

		ret, thresh2 = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
		_, contours, _= cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for i, c in enumerate(contours):
			area = cv2.contourArea(c)
			if area > 1000:
				cv2.drawContours(frame,contours,-1,(0,255,0),-1)
#				cv2.drawContours(frame, contours, i, (255, 0, 0), 3)

		cv2.imshow("frame", frame)
		#cv2.imshow("diff", diff)
		#cv2.imshow("thresh", thresh)
		#cv2.imshow("blurred", blurred)
		#cv2.imshow("thresh2", thresh2)
        
		key = cv2.waitKey(1) & 0xFF

		# if the 'q' key is pressed, stop the loop
		if key == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()