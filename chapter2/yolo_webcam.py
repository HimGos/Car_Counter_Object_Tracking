from ultralytics import YOLO
import cv2
import cvzone

# cap = cv2.VideoCapture(0)    # For Webcam
# cap.set(3, 640)
# cap.set(4, 480)

cap = cv2.VideoCapture('videos/highway.mp4')       # For videos

model = YOLO('yolo_weights/yolov8m.pt')

classnames = ["person", "bicycle", 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
              'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
              'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
              'refrigerator', 'book', 'clock', 'vase', 'scissor', 'teddy bear', 'hair drier', 'toothbrush']


while True:
    success, img = cap.read()
    results = model(img, stream=True)    # Stream uses generators
    for r in results:
        boxes = r.boxes
        for box in boxes:
            ### OPEN CV bounding box ###
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ### CVZONE Bounding box###
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Finding confidence
            conf = round(float(box.conf[0]),2)

            # Class Name
            cls = classnames[int(box.cls[0])]

            # Creating text rectangle on bbox
            cvzone.putTextRect(img=img,
                               text=f'{cls} {conf}',
                               pos=(max(0, x1), max(35, y1)),
                               scale=1,
                               thickness=1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

