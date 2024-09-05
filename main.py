# Import the necessary library
from ultralytics import YOLO
from datetime import datetime

# Load a YOLOv8n model pre-trained on COCO dataset (for transfer learning)
model = YOLO('yolov8n.pt')

# Train the model on your custom dataset
# The data.yaml file should specify the dataset details like paths to train, val sets, and class names
model.train(data='data.yaml', epochs=50, batch=16, imgsz=640)

# Get the current date and time
current_time = datetime.now()

# Format the date and time as a string
formatted_time = current_time.strftime("%Y%m%d_%H%M%S")

# Create the filename using the formatted date and time
filename = f"custom_yolov8n_trained_{formatted_time}.pt"

# Save the model after training
model.save(filename)

# Optionally, evaluate the model performance
results = model.val()

# Export the model to ONNX or other formats if needed
model.export(format='onnx')
