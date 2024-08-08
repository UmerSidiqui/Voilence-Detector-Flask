from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)

model = YOLO('yolov8n_retrained_final.pt')
video_dir = 'static/videos'
os.makedirs(video_dir, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                file_path = os.path.join(video_dir, file.filename)
                file.save(file_path)
                return redirect(url_for('stream_processing', filename=file.filename))
    return render_template('index.html')

@app.route('/stream_processing/<filename>')
def stream_processing(filename):
    return render_template('stream.html', filename=filename)

import logging

logging.basicConfig(level=logging.DEBUG)

def generate_frames(filename):
    video_path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        yield "Error: Could not open video."
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame with a confidence threshold
        results = model(frame, conf=0.5)
        annotated_frame = results[0].plot()

        # Check if violence is detected in the current frame
        violence_detected = False
        for obj in results[0].boxes:
            class_index = int(obj.cls.item())
            class_name = results[0].names[class_index]
            if class_name == 'violence':
                violence_detected = True
                break

        # Encode the frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        # Yield each frame as a server-sent event
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.1)  # Adjust as needed for processing speed

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    return Response(generate_frames(filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
