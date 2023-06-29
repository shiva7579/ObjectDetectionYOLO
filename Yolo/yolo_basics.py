from ultralytics import YOLO
import cv2
image= cv2.imread("Images/2.png")
img=cv2.resize(image,(1200,700))
model = YOLO("../YOLOweights/yolov8l.pt")
results=model(img,show = True)

cv2.waitKey()