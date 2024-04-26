import torch
from PIL import Image
import requests

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define a function for object detection
def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)
    # Perform inference
    results = model(img)
    # Display results
    results.show()

# Example usage

image_path = "image.jpg"

# Download image
# response = requests.get(image_url)
# with open(image_path, "wb") as f:
#     f.write(response.content)

# Perform object detection
detect_objects(image_path)
