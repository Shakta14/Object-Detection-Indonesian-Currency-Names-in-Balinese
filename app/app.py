from flask import Flask, request, render_template, redirect, url_for, Response, jsonify
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load custom YOLOv8 model
model = YOLO('best.pt')  # Replace with your custom model path

def detect_objects(image):
    logging.debug("Detecting objects in the image.")
    results = model(image)
    logging.debug(f"Results: {results}")
    result_image = results[0].plot()  # Use plot() to get the image with bounding boxes
    detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]  # List of detected object names
    logging.debug(f"Detected objects: {detected_objects}")
    return result_image, detected_objects

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.warning("No file part in the request.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            logging.warning("No selected file.")
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            logging.info(f"Saving file to {file_path}")
            file.save(file_path)
            
            # Load image
            image = Image.open(file_path)
            image = np.array(image)
            
            # Detect objects
            output_image, detected_objects = detect_objects(image)
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + file.filename)
            logging.info(f"Saving detected image to {output_image_path}")
            Image.fromarray(output_image).save(output_image_path)
            
            return render_template('index.html', uploaded_image=file.filename, detected_image='detected_' + file.filename, detected_objects=detected_objects)
    
    return render_template('index.html')

def gen_frames():
    camera_index = 0
    logging.debug(f"Trying to open camera at index {camera_index}")
    camera = cv2.VideoCapture(camera_index)  # Coba ganti 0 dengan 1 atau indeks lain jika perlu
    if not camera.isOpened():
        logging.error("Camera could not be opened.")
        return
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Frame could not be read.")
            break
        else:
            # Detect objects
            detected_frame, detected_objects = detect_objects(frame)
            logging.debug(f"Detected objects in frame: {detected_objects}")
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', detected_frame)
            if not ret:
                logging.error("Frame could not be encoded.")
                continue
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()
    logging.debug("Released the camera.")

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
