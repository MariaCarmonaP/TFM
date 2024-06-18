from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n-pose.yaml")  # build a new model from scratch

# yolov8m-pose, yolov8l-pose, yolov8x-pose

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

#model = YOLO("runs/pose/train73/weights/best.pt")

# Use the model
model.train(data="/home/julian/Projects/Matriculas/yv8/ocr.yaml", epochs=200, imgsz = 640)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("DATASET/IMAGES/")
#results = model("malaga_noche/", save=True)
