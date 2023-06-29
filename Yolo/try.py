from ultralytics import YOLO
import cv2
image= cv2.imread("Images/1.png")
img=cv2.resize(image,(900,700))
model = YOLO("../YOLOweights/yolov8n.pt")
results=model(img,show = True)

cv2.waitKey()