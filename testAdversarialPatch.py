from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # change to dest different models
# smallest model: yolo11n.pt
# largest model: yolo11x.pt
source = "./person with patch3.png" # change to test different images

# Run batched inference on a list of images
results = model(source, max_det = 1)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
   # masks = result.masks  # Masks object for segmentation masks outputs
    result.show()  # display to screen
