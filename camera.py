import cv2

class VideoCamera(object):
    def __init__(self):
        # change parameter as necessary
        self.video = cv2.VideoCapture(0) 

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
