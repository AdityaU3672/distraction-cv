import cv2
import numpy as np
import dlib
import math
vid = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
font = cv2.FONT_HERSHEY_SIMPLEX

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



while(True):
    ret, img = vid.read()
    size = img.shape
    width,height,pixels=img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # IF NO FACES
    if faceIndex(rects)==-1:
        img = cv2.putText(img,"No face detected ",(50,50), font, 
                1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    rect=rects[faceIndex(rects)]
    shape = predictor(gray, rect)
    image_points = make2d(shape)
    model_points=make3d()

    # SETTING UP CAMERA DETAILS
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    
    # ACTUALLY SOLVING PNP
    dist_coeffs = np.zeros((4,1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

    # GETTING THE END POINT OF THE LINE
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # DRAWING THE LINE
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    result=str(p2[0])+" "+str(p2[1])
    cv2.line(img, p1, p2, (255,0,0), 2)

    # Calculate Euler angles
    ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
    euler_angle_str = 'Pitch:{}, Yaw:{}, Roll:{}'.format(pitch, yaw, roll)
    img = cv2.putText(img, euler_angle_str,(50,50), font, 
                1, (0,255,0), 2, cv2.LINE_AA)
    
    if yaw<-30:
        cv2.putText(img,"Left",(150,50), font, 
                1, (0,0,255), 2, cv2.LINE_AA)
    elif yaw> 35:
        cv2.putText(img,"Right",(150,50), font, 
                1, (0,0,255), 2, cv2.LINE_AA)
    elif roll<-30:
        cv2.putText(img,"Tilt Right",(150,250), font, 
                1, (255,0,0), 2, cv2.LINE_AA)
    elif roll> 30:
        cv2.putText(img,"Tilt Left",(150,250), font, 
                1, (255,0,0), 2, cv2.LINE_AA)
    elif 0 < pitch < 160:
        cv2.putText(img,"Up",(150,250), font, 
                1, (255,0,0), 2, cv2.LINE_AA)
    elif -165 < pitch < 0:
        cv2.putText(img,"Down",(150,250), font, 
                1, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()

cv2.destroyAllWindows()