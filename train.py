from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data/Crack Segmentation.v1i.yolov8/data.yaml', epochs=50, imgsz=640)

# Save the model
model.save('model/best.pt')
