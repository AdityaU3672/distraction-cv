import cv2

def detector():
    face_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

    img = cv2.imread("pic.jpg")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=5, minNeighbors=1)

    count = 0
    ret = "Eyes open"

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        count += 1

    if (not count == 2):
        #font = cv2.FONT_HERSHEY_SIMPLEX
        ret = "Eyes closed"

        #cv2.putText(img, ret, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)

    cv2.imshow("Gray", img)

    cv2.waitKey()

    return ret
