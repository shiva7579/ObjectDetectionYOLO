import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
# webcam = cv2.VideoCapture(0)
# webcam.set(3,1000)
# webcam.set(4,1000)
webcam = cv2.VideoCapture("../Yolo/Videos/cars.mp4")

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
limits = [400, 297, 673, 297]
#Tracking

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

Model=YOLO("../YOLOweights/yolov8l.pt")
counts = []
while True:
    success , img = webcam.read()
    imgregion = cv2.bitwise_and(img, mask)
    imggraphics = cv2.imread ("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imggraphics,(0,0))
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

            if currentclass == "car" or currentclass == "bus" or currentclass == "truck" or currentclass == "motorbike"\
                and conf > 0.5 :
                #cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(40, y1)), scale=1, thickness=3,
                                 # offset=3)
                #cvzone.cornerRect(img, (x1, y1, w, h), colorR=(0, 0, 255), colorC=(255, 255, 255), l=5)
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    resultstracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(255,0,143),2)


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
        if limits[0] < cx <limits[2] and limits[1]-16 <cy< limits[1]+16:
            if counts.count(ID) == 0:
                counts.append(ID)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

        cvzone.putTextRect(img, f'Totalcounts: {len(counts)}', (20,20), scale=1, thickness=2)


    # cv2.resize(img,(10,10))
    # cv2.imshow("video",img)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break