import os 
from ultralytics import YOLO
import cv2
from sort.sort import *
from utils import get_car, read_license_plate, write_csv
import pickle

coco_model = YOLO(os.path.join('./' , 'models', 'yolov8n.pt'))
license_plate_detector = YOLO(os.path.join('./', 'models', 'license_plate_detector.pt'))
video_path = os.path.join('./', 'videos', 'cars.mp4')

motion_tracker = Sort()

cap = cv2.VideoCapture(video_path)

vehicles = [2, 3, 4, 5, 8, 9]

results = {}

frame_num = 0

while True: 

    ret, frame = cap.read()
    results[frame_num] = {}

    if ret == False:
        break

    detections_ = []
    detections = coco_model(frame)[0]
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles: 
            detections_.append([x1, y1, x2, y2, score])     

    track_ids = motion_tracker.update(np.asarray(detections_))

    licenses = license_plate_detector(frame)

    for license in licenses[0].boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = license
        license_plate = x1, y1, x2, y2

        x1_car, y1_car, x2_car, y2_car, car_id =  get_car(license_plate, track_ids)

        if car_id != -1:
                cropped_license = frame[int(y1):int(y2), int(x1):int(x2)]
                cropped_license_gray = cv2.cvtColor(cropped_license, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(cropped_license_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_text, license_score = read_license_plate(thresh)

                if license_text is not None:
                
                    results[frame_num][car_id] = {'car': {
                                              'bbox': [x1_car, y1_car, x2_car, y2_car]},
                                              'license_plate': {
                                                  'bbox': [x1, y1, x2, y2],
                                                  'text': license_text,
                                                  'bbox_score': score,
                                                  'text_score':license_score,
                                              }}   
    frame_num +=1

write_csv(results, './test.csv')

cap.release()
cv2.destroyAllWindows()