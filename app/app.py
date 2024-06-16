import os
import cv2
import logging
from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')  # Ganti dengan jalur model YOLOv8 custom Anda jika ada
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

camera_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
        results = model(img_rgb)
        results_img = results[0].plot()  # Menggunakan plot() untuk menambahkan kotak deteksi pada gambar
        detected_filename = 'detected_' + file.filename
        detected_path = os.path.join(app.config['UPLOAD_FOLDER'], detected_filename)
        cv2.imwrite(detected_path, cv2.cvtColor(results_img, cv2.COLOR_RGB2BGR))  # Simpan gambar dengan kotak deteksi
        detections = []
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            detections.append(label)
        return render_template('result.html', original=file.filename, detected=detected_filename, detections=detections)
    return redirect(url_for('index'))

@app.route('/camera')
def camera():
    global camera_active
    camera_active = True
    return render_template('camera.html')

@app.route('/stop_camera')
def stop_camera():
    global camera_active
    camera_active = False
    return redirect(url_for('index'))

def gen_frames():
    global camera_active
    camera = cv2.VideoCapture(0)
    while camera_active:
        success, frame = camera.read()
        if not success:
            logging.error("Frame could not be read.")
            break
        else:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
            results = model(img_rgb)
            results_img = results[0].plot()  # Menggunakan plot() untuk menambahkan kotak deteksi pada gambar
            frame = cv2.cvtColor(results_img, cv2.COLOR_RGB2BGR)  # Konversi kembali ke BGR untuk OpenCV
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
