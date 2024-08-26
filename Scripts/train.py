from ultralytics import YOLO  # type: ignore
from torch import cuda
# Load a model
# model = YOLO("yolov8n-pose.yaml")  # build a new model from scratch

# yolov8m-pose, yolov8l-pose, yolov8x-pose

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# model = YOLO("runs/pose/train73/weights/best.pt")

# Use the model
model.train(
    data=r"/home/maria/TFM/data/datasets/unfiltered_DATASET/cfg.yaml",
    imgsz=608,
    device="cuda:0" if cuda.is_available() else "cpu",
    seed=4,
    close_mosaic=0,
    auto_augment="",
    erasing=0.0,
    crop_fraction=1.0,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0,
    translate=0,
    scale=0,
    fliplr=0,
    mosaic=0,
    save_json=True,
)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/maria/TFM/data/datasets/unfiltered_DATASET/results")
# results = model("malaga_noche/", save=True)
