from flask import Flask, render_template, Response
import cv2
import pickle

app = Flask(__name__)

# Load the pre-trained pickle model

sign_model = pickle.load("model.pkl")
translate_model = pickle.load("model.pkl")

@app.route('/')
def index():
    return render_template('index.html')  # load template page


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Perform prediction using the model
            # Replace this line with your own prediction logic
            prediction = sign_model.predict(frame)

            # Display the prediction on the frame
            cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')  # concat frame one by one and show result b'Content-Type: image/jpeg\r\n\r\n' frame b'\r\n\r\n' b'Content-Type: image/jpeg\r\n\r\n' frame b'\r\n\r\n' b'Content-Type: image/jpeg\r\n\r\n' frame b'\r\n\r\n'



# Define a function to capture video from the laptop camera
def capture_video():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        else:
            # Perform prediction using the model
            # Replace this line with your own prediction logic
            prediction = sign_model.predict(frame)

            # Display the prediction on the frame
            cv2.putText(frame, prediction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(capture_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)