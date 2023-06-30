from ultralytics import YOLO
import cv2
import cvzone
import math

# webcam = cv2.VideoCapture(0)
# webcam.set(3,1000)
# webcam.set(4,1000)
webcam = cv2.VideoCapture("Videos/short_video.mp4")

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

Model=YOLO("../YOLOweights/yolov8l.pt")

while True:
    success , img = webcam.read()
    results = Model(img, stream= True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0))
            w,h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),colorR=(0,0,255),colorC=(255,255,255))
            # print(x1,y1,x2,y2)
            conf = math.ceil((box.conf[0]*100)/100)
            # print(confidence)
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)),scale=0.7,thickness=1)

    cv2.resize(img,(10,10))
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xff==ord(' '):
        break