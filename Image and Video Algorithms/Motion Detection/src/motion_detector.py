# import the necessary packages
import argparse
import datetime
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum contour area size")
args = vars(ap.parse_args())

# read form the webcam
camera = cv2.VideoCapture(0)
time.sleep(0.25)

# initialize the first frame in the video stream
firstFrame = None

# loop over the frames of the video
while True:

	(isFrame, frame) = cap.read()

	# If not valid frame, break operation
	if not isFrame:
		break

	# resize the frame, convert it to grayscale, and blur it
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	_, cnts, _= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the textq
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Changed"

	# draw the text and timestamp on the frame
	cv2.putText(frame, "Image Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), 
		    cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 0, 255), 1)

	# show the frame and record if the user presses a key
	cv2.imshow("Camera Feed", frame)
	##cv2.imshow("Thresh", thresh)
	##cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
