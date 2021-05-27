# TODO
# - calibrated coordinates for comparison while running loop
# - integrate eyes closed model
# - Pose Detection
# - Illumination Detection (Using Phone Detection)
# - End case function to call when they've been caught not paying attention


import math
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

#Vardhan function definitions-----------------------------START------------------------------------------------
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    #print("contouring")
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #try:
    if cnts:
        M = cv2.moments(max(cnts, key = cv2.contourArea))
        #print("Moments:, ", M, '\n')
        if M['m10'] and M['m01'] and M['m00']:
        #cnt = max(cnts, key = cv2.contourArea)
        #M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        else:
            #print("I see NO eyes")
            cx, cy = -1, -1
    #except:
    else:
        #print("I see no eyes")
        cx, cy = -1, -1

    #print("Eyes at", cx, cy)
    return cx, cy

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

font = cv2.FONT_HERSHEY_SIMPLEX

#Vardhan function definitions-----------------------------END------------------------------------------------

#Calibrating Loop-----------------------------START---------------------------------------------------------
calibrating = True
CALIBCOUNT = 60
consecutive = 0
thres = 30

rmaxx = 0
rminx = math.inf
rmaxy = 0
rminy = math.inf


lmaxx = 0
lminx = math.inf
lmaxy = 0
lminy = math.inf

avgleftx=0
avglefty=0
avgrightx=0
avgrightx=0
while calibrating:
    #print("Still Calibrating")
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    #print(len(rects))
    for rect in rects:
        cv2.putText(img, 'Adjust the threshold bar until eye-tracking begins', (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4) 
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        lx, ly = contouring(thresh[:, 0:mid], mid, img)
        rx, ry = contouring(thresh[:, mid:], mid, img, True)


        if lx > lmaxx:
            lmaxx = lx
        elif lx < lminx:
            lminx = lx
        if ly > lmaxy:
            lmaxy = ly
        elif ly < lminy:
            lminy = ly 
        
        if rx > rmaxx:
            rmaxx = rx
        elif rx < rminx:
            rminx = rx
        if ry > rmaxy:
            rmaxy = ry
        elif ry < rminy:
            rminy = ry
            
        #print(rminx, rmaxy, lminx, lmaxy)
        # print(lx, ly, rx, ry, consecutive)

        if (lx==-1 or rx==-1 or ly==-1 or ry==-1):
            rmaxx = 0
            rminx = math.inf
            rmaxy = 0
            rminy = math.inf
            #consecutive = 0
        #if  lmaxx - lminx > thres or lmaxy - lminy > thres:
            lmaxx = 0
            lminx = math.inf
            lmaxy = 0
            lminy = math.inf
            consecutive = 0
            avgleftx=0
            avglefty=0
            avgrightx=0
            avgrighty=0
            print("Calibration: Eyes Not Found")
        elif  rmaxx - rminx > thres or rmaxy - rminy > thres or lmaxx - lminx > thres or lmaxy - lminy > thres:
            rmaxx = 0
            rminx = math.inf
            rmaxy = 0
            rminy = math.inf
            #consecutive = 0
        #if  lmaxx - lminx > thres or lmaxy - lminy > thres:
            lmaxx = 0
            lminx = math.inf
            lmaxy = 0
            lminy = math.inf
            consecutive = 0
            avgleftx=0
            avglefty=0
            avgrightx=0
            avgrighty=0
            print("Large change noted")
        else:
            avgleftx+=lx
            avglefty+=ly
            avgrightx+=rx
            avgrighty+=ry
            consecutive += 1

        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)

    if consecutive > CALIBCOUNT:
        print("Congrats Calibration over")
        avgleftx/=consecutive
        avglefty/=consecutive
        avgrightx/=consecutive
        avgrighty/=consecutive
        break
    
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
        continue
#Calibrating Loop-----------------------------END---------------------------------------------------------

#Checking for eyes closed-----------------------------START------------------------------------------------------

#Gets the Eye Aspect ratio (same parameters as get mask)
def eye_aspect_ratio(shape, side):
    eye = [shape[i] for i in side]
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear
    

def end_case(eyes_closed=False, deviation=False):
    if eyes_closed:
        #output sound
        print("Eyes closed")

    elif deviation:
        #print statement
        print("Deviation exists")
        
    


#Checking for eyes closed-----------------------------END------------------------------------------------------

LEAR = 0
REAR = 0

EAR_THRES = 0.2
CLOSE_COUNT = 0
OPEN_COUNT = 0
CLOSE_THRESH = 70

DEVIATION = 20 # dynamic_deviation(resolution)

def isDeviated(base, coord, dev = DEVIATION):
    return dist.euclidean(base, coord) > dev

lbase = (avgleftx, avglefty)
rbase = (avgrightx, avgrighty)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    #print(avgleftx, avglefty,"<- LEFT, RIGHT ->", avgrightx, avgrighty)
    #break

    for rect in rects:
        
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        LEAR = eye_aspect_ratio(shape, left)
        REAR = eye_aspect_ratio(shape, right)
        AVGEAR=(LEAR+REAR)/2

        #print("Left eye EAR: ", LEAR, "Right eye EAR: ", REAR)
        
        if AVGEAR < EAR_THRES:
            CLOSE_COUNT += 1
            print("Close for ",CLOSE_COUNT, "consecutive frames")
            if CLOSE_COUNT > CLOSE_THRESH:
                #Game over function
                print("Pay Attention")
                end_case(eyes_closed=True)
            break
        
        elif CLOSE_COUNT > 15 and OPEN_COUNT<10:
            OPEN_COUNT+=1
            print("Eyes open for a limited time only, ", OPEN_COUNT)
            break

        else:
            #print("Congratulations you are woke")
            CLOSE_COUNT=0
            OPEN_COUNT=0

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        lx, ly = contouring(thresh[:, 0:mid], mid, img)
        rx, ry = contouring(thresh[:, mid:], mid, img, True)

        if isDeviated(rbase, (rx, ry)) or isDeviated(lbase, (lx, ly)):
            end_case(deviation = True)

        print(lx, ly,"<-- Left : Right -->", rx, ry, consecutive)

        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
