import cv2
import dlib
import numpy as np


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


def eye_on_mask(mask, side): #creating an eye mask
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32) 
    mask = cv2.fillConvexPoly(mask, points, 255) #fill eye region with white others black
    return mask


def contouring(thresh, mid, img, right=False): #finding pupils
    cnts, _ = cv2.findContours(
    	thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #find contours
    try:
        cnt = max(cnts, key=cv2.contourArea) #find the biggest contour
        M = cv2.moments(cnt) #find the moments of the contour
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2) #draw a circle around the pupil
    except:
        pass


detector = dlib.get_frontal_face_detector() #face detector 
predictor = dlib.shape_predictor('shape_68.dat') #landmark detector(68 point landmarks)

left = [36, 37, 38, 39, 40, 41] #left eye landmarks
right = [42, 43, 44, 45, 46, 47] #right eye landmarks

cap = cv2.VideoCapture(0) #start the webcam
ret, img = cap.read() #read the image
thresh = img.copy()

cv2.namedWindow('image') #create a window
kernel = np.ones((9, 9), np.uint8) #create a kernel


def nothing(x):
    pass


cv2.createTrackbar('threshold', 'image', 0, 255, nothing) #create a trackbar

while(True):
    ret, img = cap.read() #read the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the image to grayscale
    rects = detector(gray, 1) #detect faces
    for rect in rects: #loop over the faces

        shape = predictor(gray, rect) #find facial landmarks
        shape = shape_to_np(shape) #convert the landmarks to numpy array
        mask = np.zeros(img.shape[:2], dtype=np.uint8) #create a mask
        mask = eye_on_mask(mask, left) 
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5) #binary mask of eye regions
        eyes = cv2.bitwise_and(img, img, mask=mask) #extract eye regions
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2 #find the mid point of the eye
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY) #convert the eye regions to grayscale
        threshold = cv2.getTrackbarPos('threshold', 'image') 
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY) #apply thresholding
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3 
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img) #find the pupil
        contouring(thresh[:, mid:], mid, img, True) 
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img) #show the image
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
