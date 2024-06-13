from ultralytics import YOLO

model = YOLO('best.pt')

model.info()

results = model.predict(source='0', imgsz=640, conf=0.3, save=True, show=True)

