import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
# webcam = cv2.VideoCapture(0)
# webcam.set(3,1000)
# webcam.set(4,1000)
webcam = cv2.VideoCapture("../Yolo/Videos/people.mp4")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv2.imread("mask.png")
limitsup= [103,161,296,161]
limitsdown= [527,497,773,497]
#Tracking

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

Model=YOLO("../YOLOweights/yolov8l.pt")
countsup = []
countsdown = []
while True:
    success , img = webcam.read()
    imgregion = cv2.bitwise_and(img, mask)
    imggraphics = cv2.imread ("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imggraphics, (730,260))
    results = Model(imgregion, stream= True)
    detections = np.empty((0, 5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0))
            w,h = x2-x1,y2-y1
            # cvzone.cornerRect(img,(x1,y1,w,h),colorR=(0,0,255),colorC=(255,255,255),l=5)
            # print(x1,y1,x2,y2)
            conf = math.ceil((box.conf[0]*100)/100)
            # print(confidence)
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass == "person" and conf > 0.5 :
                #cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(40, y1)), scale=1, thickness=3,
                                 # offset=3)
                #cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(255, 255, 255), l=5)
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    resultstracker = tracker.update(detections)
    cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (255, 0, 143), 2)
    cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (255, 0, 143), 2)

    for results in resultstracker:
        x1, y1, x2, y2, ID= results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(img, f'{int(ID)}{currentclass} {conf}', (max(0, x1), max(40, y1)), scale=0.8, thickness=1,
                       offset=3)
        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(255, 255, 255), l=5)
        cx, cy = x1+w//2, y1+h//2
        # cv2.circle(img,(cx,cy),2,(255,0,231),cv2.FILLED)
        if limitsup[0] < cx < limitsup[2] and limitsup[1]-16 < cy < limitsup[1]+16:
            if countsup.count(ID) == 0:
                countsup.append(ID)
                cv2.line(img, (limitsup[0], limitsup[1]), (limitsup[2], limitsup[3]), (0, 255, 0), 2)

        if limitsdown[0] < cx <limitsdown[2] and limitsdown[1]-16 < cy < limitsdown[1]+16:
            if countsdown.count(ID) == 0:
                countsdown.append(ID)
                cv2.line(img, (limitsdown[0], limitsdown[1]), (limitsdown[2], limitsdown[3]), (0, 255, 0), 2)

        # cvzone.putTextRect(img, f'{len(countsup)}', (950,320), scale=1, thickness=2)
        # cvzone.putTextRect(img, f'{len(countsdown)}', (1200, 320), scale=1, thickness=2)
        cv2.putText(img,str(len(countsup)), (950, 350), cv2.FONT_HERSHEY_PLAIN,5,(138,245,0),3)
        cv2.putText(img, str(len(countsdown)), (1200, 350), cv2.FONT_HERSHEY_PLAIN, 5, (138, 245, 0), 3)


    # cv2.resize(img,(10,10))
    # cv2.imshow("video",img)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break