from ultralytics import YOLO
import numpy
import cv2
import cvzone
import math

classnames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

vid = cv2.VideoCapture("../Yolo/Videos/ppe-3.mp4")
model = YOLO("ppe.pt")
while True:
    success, img = vid.read()
    final = model(img, stream = True)
    for r in final:
        kkkk = r.boxes
        for k in kkkk:
            x1, y1, x2, y2 = k.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((k.conf[0]*100))/100
            cls = int(k.cls[0])
            classn = classnames[cls]
            if classn == "Hardhat" or classn == "Mask" or classn == "Safety Vest" and conf > 0.3:
                color = (0,255,0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img, f"{classn}{conf}", (max(0, x1), max(40, y1)), 1, 1, color)
            elif classn == 'NO-Hardhat' or classn == "NO-Mask" or classn == "NO-Safety Vest" and conf > 0.3:
                color = (0,0,255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img, f"{classn}{conf}", (max(0, x1), max(40, y1)), 1, 1, color)
            elif classn == "Person" and conf > 0.3:
                color = (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img, f"{classn}{conf}", (max(0, x1), max(40, y1)), 1, 1, color)
            else:
                color = [0, 0, 0]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cvzone.putTextRect(img, f"{classn}{conf}", (max(0, x1), max(40, y1)), 1, 1, color)
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xff == ord('m'):
        break



