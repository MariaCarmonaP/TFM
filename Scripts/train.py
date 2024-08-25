from ultralytics import YOLO  # type: ignore

# Load a model
# model = YOLO("yolov8n-pose.yaml")  # build a new model from scratch

# yolov8m-pose, yolov8l-pose, yolov8x-pose

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# model = YOLO("runs/pose/train73/weights/best.pt")

# Use the model
model.train(
    data=r"C:\Users\sierr\Documents\Uni\TFM\TFM\Data\Datasets\DATASET_v1\cfg.yaml",
    epochs=100,
    imgsz=640,
)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("DATASET/IMAGES/")
# results = model("malaga_noche/", save=True)
