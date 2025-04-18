import cv2
import time
from ultralytics import YOLO
#from ultralytics import RTDETR
import torch
torch.cuda.set_device(0)

def yolo_inference(frame, model, image_size, conf_threshold):
    results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, iou = 0.25)
    frame =results[0].plot()
    return frame

def main():
    image_size = 640
    conf_threshold = 0.5
    #model = YOLOv10("yolov3-sppu.pt")
    model = YOLO("yolo11n.pt")
    # Load a COCO-pretrained RT-DETR-l model
    #model = RTDETR("weights/rtdetr-l.pt")
    #model = YOLO("yolov8n.pt")

    source = 2
    #source = "tests/CARROS.mp4"
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Eroor: Could not open camera.")
        return
    while True:
        start_time = time.time()
        ret, frmae = cap.read()
        if not ret:
            break
        frame = yolo_inference(frmae, model, image_size, conf_threshold)
        end_time = time.time()
        fps = 1/ (end_time - start_time)
        framefps = "FPS: {:.2f}".format(fps)
        cv2.rectangle(frame, (10,1), (120,20), (0, 0,0), -1)
        cv2.putText(frame, framefps, (15,17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("YOLOv11", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
main()