import flask
from flask import Flask, request, jsonify, render_template
from combined_0 import *

up = Flask(__name__)

@up.route('/')
def home():
    ## Place While Loops Here

    return render_template("index.html")

if __name__=='__main__':

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('resc/shape_68.dat')
    cap = cv2.VideoCapture(2)
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
    base_yaw = 0
    base_pitch = 0


    while True:
        _, img = cap.read()
        #new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray )# , 1) # adding this second argument detects faces better, but is significantyl slower
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
                cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
                shape = predictor(gray, face)
                ret, pitch, yaw, roll = updt_pose(shape)
                pitch = 180 - pitch if pitch > 0 else -180 - pitch
                pose_str = "Pitch:{}, Yaw:{}, Roll:{}".format(pitch, yaw, roll)
                put_text(pose_str, (25, 80), (0,255,0), img)
                
                base_yaw = calib_hori.update(yaw)
                base_pitch = calib_vert.update(pitch )
                
                cv2.imshow("Output", img)

                if not (base_yaw == None or base_pitch == None):
                    put_text("FREE TONIGHT? - You got snap?", (25, 40), (0,255,0))
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    
    #for _ in range(3):
        #put_text("FREE TONIGHT? - You got snap?", (25, 40), (0,255,0))

    #Main Loop
    while True:
        print(base_yaw, base_pitch)
        _, img = cap.read()
        #new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray )#, 1) # adding this second argument detects faces better, but is significantyl slower
        biggestface = faceIndex(faces)

        if biggestface < 0:
            put_text("FACE NOT FOUND", (25, 40), (0,255,0))
        else:
            face = faces[biggestface]
            shape = predictor(gray, face)
            shape_np = shape_to_np(shape)

            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

            ret, pitch, yaw, roll = updt_pose(shape)

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

    up.run(host='127.0.0.1',port='3000',debug=True)

