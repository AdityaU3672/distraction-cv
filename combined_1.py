# TODO
# implement counter class


import numpy as np
import math
import dlib, cv2
from scipy.spatial import distance as dist
modelFile = "DNN/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "DNN/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

#----------- Supplementary Function Definitions -----------------

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

def updt_gaze(landmarks):
        #change to left and right (the parameter that is)
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        
        return (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2 

# FUNCTION TO MAKE 2D IMAGE
def make2d(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)
# FUNCTION DEFINITION END 

# FUNCTION TO MAKE 3D MODEL POINTS
def make3d():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)
# FUNCTION DEFINITION END 

# GETTING THE EULER ANGLES
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
	 # Unit conversion: convert radians to degrees
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    return 0, Y, X, Z
#FUNCTION DEFINITION END

# CHOOSING THE LARGEST FACE
def faceIndex(rects):
    if len(rects)==1:
        return 0
    elif len(rects)==0:
        return -1
    area=((rect.right()-rect.left())*(rect.bottom()-rect.top()) for rect in rects)
    area=list(area)
    maxIndex=0
    maximum=area[0]
    for i in range(1,len(area)):
        if (area[i]>maximum):
            maxIndex=i
            maximum=area[i]
    return maxIndex
#FUNCTION DEFINITION END

def updt_pose(shape):
        image_points = make2d(shape)

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # DRAWING THE LINE
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        result=str(p2[0])+" "+str(p2[1])
        cv2.line(img, p1, p2, (255,0,0), 2)

        # Calculate Euler angles
        return get_euler_angle(rotation_vector)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def put_text(inpt, loc, clr = (0, 0, 255)):
    return cv2.putText(img, inpt, loc, font, 1, clr, 2, cv2.LINE_AA)

def eye_aspect_ratio(shape, side):
    eye = [shape[i] for i in side]
    return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))

# could be optimized better both in terms of abstraction barriers and actual implamentation

class Counter():
    def __init__(self, frames, ratio = 0.6) -> None:
        self.thres = int(frames * ratio)
        self.frames = frames
        self.count0 = 0
        self.count1 = 0

    def update(self, cond_0, cond_1 = False):
        if cond_0:
            if self.count0 < self.frames:
                self.count0 += 1
            if self.count1 > 0:
                self.count1 -= 1
        elif cond_1:
            if self.count1 < self.frames:
                self.count1 += 1
            if self.count0 > 0:
                self.count0 -= 1
        else:
            self.decrement()
        
        if (self.count0 > self.thres):
            return 1
        else:
            return -(self.count1 > self.thres)

    def decrement(self):
        if self.count1 > 0:
            self.count1 -= 1
        if self.count0 > 0:
            self.count0 -= 1

    def display(self, labe = ""):
        print(labe, self.thres, self.frames, self.count0, self.count1)

    def reset(self) -> None:
        self.count0 = 0
        self.count1 = 0


def check_pose(pitch,yaw,roll):
    pose_str = "Pitch:{}, Yaw:{}, Roll:{}".format(pitch, yaw, roll)
    put_text(pose_str, (25, 80), (0,255,0))
    
    consec_hori.display("Horizontal:")
    consec_vert.display("Vertical:")
        
    return consec_hori.update(yaw<-30, yaw > 35), consec_vert.update(pitch > 13, -10 > pitch)

def check_eyes(ear_avg, shape):

    if ear_avg > 0.2:
        gaze_ratio = updt_gaze(shape)
        gazedir = consec_gaze.update(gaze_ratio >= 1.5, gaze_ratio <= 1)
        consec_gaze.display("Gaze:")

    else:
        gazedir = 2
        consec_gaze.decrement()

    return gazedir

def end_case(horizontal, vertical, gaze,):
    if (horizontal or vertical):
        print("Look straight.Your horizontal and vertical viewpoints are distorted.")
        #make_noise(5)

    if (gaze==2):
        print("Wake up. You are driving.")
        #make_noise(10)
    
    # call make_noise()

def make_noise(volume, file):
    #implementation specific
    pass


def not_in(tLeft, bRight, xmin, xmax, ymin, ymax):
    return (tLeft[0]<xmin or tLeft[1]<ymin or bRight[0]>xmax or bRight[1]>ymax)
    

#def calibration(
    

class Calibrator:
    def __init__(self, limt, thres, namee = "") -> None:
        self.limt = limt
        self.thres = thres
        self.consec = 0
        self.min = math.inf
        self.max = -math.inf
        self.sum = 0
        self.namee = namee

    def update(self, val):
        
        self.sum+=val
        if val > self.max:
            self.max = val
        
        elif val < self.min:
            self.min = val
        
        if self.max-self.min >= self.thres:
            print(self.namee, "Threshold Exceeded")
            self.reset()
            
        self.consec += 1  
        if self.consec >= self.limt:
            return self.sum / self.consec

        return None

    def display(self, labl = ""):
        print(labl, self.consec, self.sum, self.min, self.max)

    def reset(self):
        self.consec = 0  
        self.sum=0 
        self.min = math.inf
        self.max = -math.inf


#  CAFFE DNN
def dnn_detect(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    outs = []
    
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
            
            outs.append((x, y, x1, y1))
        
    return outs # faces.shape[2][outs]


def to_rectangle(arr):
    return list(map(lambda x: dlib.rectangle(left = x[0], top = x[1], right = x[2], bottom = x[3]), arr))


#----------- Actually Running It -----------------

if __name__ == "__main__":

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_68.dat')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    model_points = make3d()
    dist_coeffs = np.zeros((4,1))
    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]
    gaze_ratio = 1
    calibrating = True

    calib_vert = Calibrator(100, 10, "verti")
    calib_hori = Calibrator(100, 25, "hori")

    consec_gaze = Counter(70)
    consec_hori = Counter(25)
    consec_vert = Counter(25)
    consec_attn = Counter(100)

    Vertpt = {0 : "CENTER", 1 : "UP", -1 : "DOWN"} # Looking up and down (pitch)
    Hoript = {0 : "CENTER", 1 : "LEFT", -1 : "RIGHT"} # Looking left and right (yaw)
    Gazept = {0 : "CENTER", 1 : "LEFT", -1 : "RIGHT", 2 : "CLOSED"} #Gaze

    # assuming that the camera stays constant, we can get these values at the start
    # if there is the possibility that the camera can change, put everythinb below this comment into the while loop
    _, img = cap.read()
    size = img.shape
    #height,width,pixels=img.shape
    
    xmin = (size[1]//10)
    xmax = xmin * 9
    ymin = (size[0]//10)
    ymax = ymin * 9

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )

    # calibration loop
    while True:
        _, img = cap.read()
        #new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #-----

        #faces = detector(gray )#, 1) # adding this second argument detects faces better, but is significantly slower
        
        faces = dnn_detect(img)
        print(faces)

        faces = to_rectangle(faces)

        #----
        biggestface = faceIndex(faces)

        calib_hori.display()
        calib_vert.display()

        if biggestface < 0:
            put_text("FACE NOT FOUND", (25, 40), (0,255,0))
            print("Face not Found")
            calib_hori.reset()
            calib_vert.reset()
        
        else:
            face = faces[biggestface]
            #print((face.left(), face.top()), (face.right(), face.bottom()))
            #print(xmin,ymin,xmax,ymax)
            if not_in((face.left(), face.top()), (face.right(), face.bottom()), xmin, xmax, ymin, ymax):
                put_text("CENTER FACE IN FRAME", (25, 40), (0,255,0))
                calib_hori.reset()
                calib_vert.reset()
                print("Out of Frame")

            else:
                shape = predictor(gray, face)
                ret, pitch, yaw, roll = updt_pose(shape)
                pitch = 180 - pitch if pitch > 0 else -180 - pitch
                pose_str = "Pitch:{}, Yaw:{}, Roll:{}".format(pitch, yaw, roll)
                put_text(pose_str, (25, 80), (0,255,0))
                
                base_yaw = calib_hori.update(yaw)
                base_pitch = calib_vert.update(pitch)
                
                cv2.imshow("Output", img)

                if not (base_yaw == None or base_pitch == None):
                    put_text("FREE TONIGHT? - You got snap?", (25, 40), (0,255,0))
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    
            for _ in range(3):
                put_text("FREE TONIGHT? - You got snap?", (25, 40), (0,255,0))

    #Main Loop
    while True:
        print(base_yaw, base_pitch)
        _, img = cap.read()
        #new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #-----

        #faces = detector(gray )#, 1) # adding this second argument detects faces better, but is significantly slower
        
        faces = dnn_detect(img)
        print(faces)

        faces = to_rectangle(faces)

        #----

        biggestface = faceIndex(faces)

        if biggestface < 0:
            put_text("FACE NOT FOUND", (25, 40), (0,255,0))
        else:
            face = faces[biggestface]
            shape = predictor(gray, face)
            shape_np = shape_to_np(shape)

            ret, pitch, yaw, roll = updt_pose(shape)
            pitch = 180 - pitch if pitch > 0 else -180 - pitch

            pitch -= base_pitch
            yaw -= base_yaw

            ear_left = eye_aspect_ratio(shape_np, left)
            ear_right = eye_aspect_ratio(shape_np, right)
            ear_avg = (ear_left + ear_right)/2

            gaze_str = "EAR:{:.2f}, Gaze:{:.2f}".format(ear_avg, gaze_ratio)    
            put_text(gaze_str, (25, 40), (0,255,0))
            Horizontal, Vertical = check_pose(pitch,yaw,roll)
            
            if not (Vertical or Horizontal):
                Gaze = check_eyes(ear_avg, shape)
                put_text("GAZE: " + Gazept[Gaze] , (25, 150))

            put_text("HORI: " + Hoript[Horizontal], (25, 190))
            put_text("VERT: " + Vertpt[Vertical], (25, 230))
                
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()