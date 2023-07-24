## Importing libraries
from ultralytics import YOLO
import cv2
import cvzone
from sort import Sort
import numpy as np

# Creating video capture object
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

# Masking, used for masking video to focus on specific region
mask = cv2.imread('car_counter/mask.png')

# Initializing the object tracker
tracker = Sort(max_age=20,      # Number of frames it wait for object to be seen again
               min_hits=3,
               iou_threshold=0.3)

# Coordinates of a line used for counting objects
limits = [500, 350, 700, 350]

# List to store the unique IDs of objects that have passed through the counting line
total_counts = []

while True:
    success, img = cap.read()

    # Focus on specific regions of the frame using the provided mask.
    img_region = cv2.bitwise_and(img, mask)

    results = model(img_region, stream=True)    # Stream uses generators

    # Top left counter image or annotations to be overlayed on the frame.
    img_graphics = cv2.imread("car_counter/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(imgBack=img, imgFront=img_graphics, pos=(0, 0))

    # Store the bounding box coordinates and ID of detected objects.
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
                
                # Bounding box and confidence is added to the detections array.
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    ## OBJECT TRACKING
    # Detected object information from detections is passed to the tracker, 
    #   and it updates the tracks for each object.
    result_tracker = tracker.update(detections)

    # Creating a counting line
    cv2.line(img=img,
             pt1=(limits[0], limits[1]),      # From which point
             pt2=(limits[2], limits[3]),      # Till which point
             color=(0, 0, 225), 
             thickness=5)

    # Tracking the detected object
    for results in result_tracker:

        # For each tracked object, extracts the bounding box coordinates & ID.
        x1, y1, x2, y2, Id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1

        # Creating box only on tracked objects, others ignored
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
        
        # Creating centre point within box which will be counted when touches line
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        # checks if the centroid of the object crosses the counting line
        if limits[0]< cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            
            # checks if the onject ID is not already in the total_counts list
            if total_counts.count(Id) == 0:
                # object just crossed the line for the first time, so the object ID is added to total_counts.
                total_counts.append(Id)
                # Line color also changes when we touch it
                cv2.line(img,(limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, ), thickness=5 )


    # Showing count
    # cvzone.putTextRect(img=img, text=f'Count: {len(total_counts)}', pos=(50, 50),
    #                     colorT=(0, 0, 0), colorR=(0, 255, 255), colorB=(255, 255, 0))

    # Showing fancy count at top left corner
    cv2.putText(img=img, text=str(len(total_counts)), org=(255, 100), 
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(50, 50, 255), thickness=8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", img_region)
    cv2.waitKey(1)
