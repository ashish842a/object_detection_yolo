import torch
from PIL import Image
import requests
import cv2
import numpy as np
from io import BytesIO

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define a function for object detection
def detect_objects(image_path):
    try:
        # Load image
        img = Image.open(image_path)
        # Perform inference
        results = model(img)
        # Display results
        results.show()
    except Exception as e:
        print("Error occurred:", e)

# Define a function for video detection
def detect_objects_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, (640, 480))  # You can change the dimensions as needed
            # Convert frame to PIL Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Perform inference
            results = model(img)
            # Convert results to numpy array
            output_img = np.asarray(results.render()[0])
            # Convert output image back to BGR format
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            # Display output
            cv2.imshow('YOLOv5 Object Detection', output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error occurred:", e)

# Example usage for image detection
image_path = "image.jpg"
detect_objects(image_path)

# Example usage for video detection
video_path = "https://videos.pexels.com/video-files/15921892/15921892-uhd_3840_2160_50fps.mp4"
detect_objects_video(video_path)
