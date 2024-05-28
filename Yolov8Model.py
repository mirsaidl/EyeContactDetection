from ultralytics import YOLO

model = YOLO('models/yolov8/train/weights/best.pt', device='cuda')
model.to('cuda')
classes = {0: 'Back',
 1: 'Front',
 2: 'FrontLeft',
 3: 'FrontRight',
 4: 'Unused',
 5: 'Left',
 6: 'Ignore',
 7: 'Right'}


def detect_face_direction(face_region):
    results = model.predict(face_region, imgsz=224, conf=0.5, max_det=1, classes=[0,1,2,3,5,7])
    
    try:
        pred = classes[int(results[0].boxes.cls)]
        conf = float(results[0].boxes.conf)
        boxes = results[0].boxes.xywh
    except:
        return 'NoDetection'
    
    return pred