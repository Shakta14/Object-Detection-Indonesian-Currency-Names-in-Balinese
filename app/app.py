import logging
from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Muat model YOLOv8 kustom Anda
model = YOLO('best.pt')  # Ganti dengan path model kustom Anda

def detect_objects(image):
    logging.debug("Mendeteksi objek dalam gambar.")
    
    # Pastikan gambar dalam format yang benar
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    elif not isinstance(image, Image.Image):
        logging.error("Format gambar tidak didukung.")
        return np.array([]), []
    
    # Konversi gambar ke array numpy
    image_np = np.array(image)
    
    # Debugging: cek bentuk gambar
    logging.debug(f"Bentuk gambar: {image_np.shape}")
    
    # Lakukan inferensi
    results = model(image_np)
    if not results:
        logging.error("Tidak ada hasil dari inferensi model.")
        return image_np, []
    
    # Jika tidak ada deteksi, log peringatan
    if results[0].boxes is None or len(results[0].boxes) == 0:
        logging.warning("Tidak ada objek yang terdeteksi.")
        return image_np, []
    
    result_image = results[0].plot()  # Gunakan plot() untuk mendapatkan gambar dengan kotak pembatas
    detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]  # Daftar nama objek yang terdeteksi
    
    # Debugging: log objek yang terdeteksi
    logging.debug(f"Objek yang terdeteksi: {detected_objects}")
    
    return result_image, detected_objects

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            logging.warning("Tidak ada bagian file dalam permintaan.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            logging.warning("Tidak ada file yang dipilih.")
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            logging.info(f"Menyimpan file ke {file_path}")
            file.save(file_path)
            
            # Muat gambar
            image = Image.open(file_path)
            image = np.array(image)
            
            # Deteksi objek
            output_image, detected_objects = detect_objects(image)
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + file.filename)
            logging.info(f"Menyimpan gambar yang terdeteksi ke {output_image_path}")
            Image.fromarray(output_image).save(output_image_path)
            
            return redirect(url_for('results', filename=file.filename, detected_filename='detected_' + file.filename, objects=','.join(detected_objects)))
    
    return render_template('index.html')

@app.route('/results')
def results():
    filename = request.args.get('filename')
    detected_filename = request.args.get('detected_filename')
    objects = request.args.get('objects').split(',')
    return render_template('results.html', filename=filename, detected_filename=detected_filename, objects=objects)

def gen_frames():
    camera_index = 1
    logging.debug(f"Mencoba membuka kamera di indeks {camera_index}")
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        logging.error("Kamera tidak bisa dibuka.")
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Frame tidak bisa dibaca.")
            break
        else:
            logging.debug("Frame berhasil dibaca.")
            
            # Konversi frame ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logging.debug("Frame berhasil dikonversi ke RGB.")
            
            # Deteksi objek
            detected_frame, detected_objects = detect_objects(frame_rgb)
            logging.debug(f"Objek yang terdeteksi dalam frame: {detected_objects}")
            
            if detected_frame is None or not isinstance(detected_frame, np.ndarray):
                logging.error("Frame yang terdeteksi tidak valid.")
                continue
            
            # Encode frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', detected_frame)
            if not ret:
                logging.error("Frame tidak bisa dienkode.")
                continue
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()
    logging.debug("Kamera dilepas.")

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
