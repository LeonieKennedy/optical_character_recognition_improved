from ultralytics import YOLO

# model architecture
model = YOLO("yolov8n.py")

# train the model with custom dataset 
#   - epochs determined through experimenting
#   - patience determines how many epochs to wait with the same precision before stopping
model.train(data="data.yaml", epochs=696, patience=300)


