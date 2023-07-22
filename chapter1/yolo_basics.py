from ultralytics import YOLO
import cv2


model = YOLO('../yolo_weights/yolov8l.pt')
results = model("chapter1/images/1.png", show=True)
cv2.waitKey(0)       # Image doesnt disappear using it
