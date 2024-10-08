import cv2
import numpy as np
from ultralytics import YOLO

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.5

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


yolo_model = "yolo_phones.pt"
# Load YOLOv8 model
model = YOLO(rf"/home/javier/Documents/python_files/kalman_filter/kalman_yolo/models/best.pt")
model.fuse()

# Define detection parameters
conf_thres = 0.20
iou_thres = 0.10
max_det = 10
classes = (list(range(1)))
device = "cpu"

# Load video 
video = cv2.VideoCapture("kalman_filter/kalman_yolo/phone.mp4")

# Load Kalman filter 
kf = KalmanFilter()

# iterate through each frame
while True:
    # Load frame from video
    ret,frame = video.read()

    # Enhance color Saturation
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:,:,1] = frame[:,:,1] * 1.6
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

    # Make prediction over frame
    results = model.track(  source=frame,
                            conf=conf_thres,
                            max_det=max_det,
                            classes=classes,
                            device=device,
                            verbose=False,
                            persist=True,
                            retina_masks=True)
    
    # Plot Values
    plotted_image = results[0].plot()
    # Get prediction values
    bounding_box = results[0].boxes.xywh
    classes = results[0].boxes.cls

    # Get bbox data of the detection results 
    for bbox,cls in zip(bounding_box,classes):
        print(f"{results[0].names.get(int(cls))}: [x: {bbox[0]}, y: {bbox[1]}]")
        x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
    
    
    y_1 = int(int(y)-(h/2))
    y_2 = int(int(y)+(h/2))

    x_1 = int(int(x)-(w/2))
    x_2 = int(int(x)+(w/2))

    # plotted_image = plotted_image[y_1:y_2,x_1:x_2,:]

    # Get prediction
    predicted = kf.predict(int(x),int(y))
    
    # Draw a dot with the detected coordinates 
    cv2.circle(plotted_image,(predicted[0],predicted[1]),10,(255,0,0),3)
    cv2.circle(plotted_image,(int(x),int(y)),5,(0,255,0),3)

    # Show image with detected centroid
    cv2.imshow("frame",plotted_image)
    cv2.waitKey(25)

