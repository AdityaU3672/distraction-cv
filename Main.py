import flask
from flask import Flask, request, jsonify, render_template, Response
import droid_eyes as DE
#from camera import Camera
#from refactored import *

up = Flask(__name__)

@up.route('/')
def index():
    ## Place While Loops Here
    return render_template("index.html")

def gen(camera):
    while True:
        _, _, _ = camera.update()
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@up.route('/video_feed')
def video_feed():
    camera = DE.attn_detector("resc/shape_68.dat")

    
    return Response(gen(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=='__main__':
    up.run(host='127.0.0.1',port='3000',debug=True)
