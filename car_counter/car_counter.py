from ultralytics import YOLO
import cv2
import cvzone
from sort import *


cap = cv2.VideoCapture('videos/highway.mp4')       # For videos

model = YOLO('yolo_weights/yolov8l.pt')

classnames = ["person", "bicycle", 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
              'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissor', 'teddy bear', 'hair drier', 'toothbrush']

# Masking
mask = cv2.imread('car_counter/mask.png')

# Tracking
tracker = Sort(max_age=20,      # Number of frames it wait for object to be seen again
               min_hits=3,
               iou_threshold=0.3)

# Creating line for counting
limits = [500, 350, 700, 350]

# Creating counter
total_counts = []

while True:
    success, img = cap.read()
    # Overlapping mask over image
    img_region = cv2.bitwise_and(img, mask)

    results = model(img_region, stream=True)    # Stream uses generators

    detections = np.empty((0, 5))             # x1,y1,x2,y2,id_number    

    for r in results:
        boxes = r.boxes
        for box in boxes:
            ### OPEN CV bounding box ###
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ### CVZONE Bounding box###
            w, h = x2-x1, y2-y1
            

            # Finding confidence
            conf = round(float(box.conf[0]),2)

            # Class Name
            cls = classnames[int(box.cls[0])]

            if (cls=='bus' or cls=='car') and conf > 0.5:
                
                ## COMMENTED IT COZ WE FOCUS ON TRACKING RATHER DETECTING
                # cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=10, rt=5)

                # # Creating text rectangle on bbox
                # cvzone.putTextRect(img=img,
                #                 text=f'{cls} {conf}',
                #                 pos=(max(0, x1), max(35, y1)),
                #                 scale=1,
                #                 thickness=1,
                #                 offset=5,
                #                 colorT=(0, 0, 0),
                #                 colorR=(0, 255, 255),
                #                 colorB=(255, 255, 0))
                
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    result_tracker = tracker.update(detections)
    # Line default color
    cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 225), thickness=5 )

    # Tracking the detected object
    for results in result_tracker:
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        cvzone.cornerRect(img=img, bbox=(x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img=img,
                                text=f'{int(Id)}',
                                pos=(max(0, x1), max(35, y1)),
                                scale=2,
                                thickness=3,
                                offset=10,
                                colorT=(0, 0, 0),
                                colorR=(0, 255, 255),
                                colorB=(255, 255, 0))
        
        # Creating centre point which will be counted when touches line
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        if limits[0]< cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
             if total_counts.count(Id) == 0:
                total_counts.append(Id)
                # Line color also changes when we touch it
                cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, ), thickness=5 )


    # Showing count
    cvzone.putTextRect(img=img,
                        text=f'Count: {len(total_counts)}',
                        pos=(50, 50),
                        colorT=(0, 0, 0),
                        colorR=(0, 255, 255),
                        colorB=(255, 255, 0))

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", img_region)
    cv2.waitKey(1)
