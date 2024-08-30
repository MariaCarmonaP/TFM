"""Basic training script"""
import time
from ultralytics import YOLO  # type: ignore
from torch import cuda

# Load a model
# model = YOLO("yolov8n-pose.yaml")  # build a new model from scratch

# yolov8m-pose, yolov8l-pose, yolov8x-pose
optimizers = ["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"]
for optimizer in optimizers:
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # model = YOLO("runs/pose/train73/weights/best.pt")
    start_time = time.time()
    # Use the model
    model.train(
        data=r"/home/maria/TFM/data/datasets/filtered_DATASET_v2/cfg.yaml",
        project=r"/home/maria/TFM/data/results/filtered_DATASET_v2/",
        name=optimizer,
        epochs=40,
        patience=5,
        imgsz=608,
        device="cuda:0" if cuda.is_available() else "cpu",
        seed=4,
        optimizer=optimizer,
        cos_lr=True,
        lr0=0.0001,
        cls=0.6,
        close_mosaic=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        translate=0,
        scale=0,
        fliplr=0,
        mosaic=0,
    )  # train the model

    end_time = time.time()
    print("Time: ", end_time - start_time)
    with open(f"/home/maria/TFM/data/results/filtered_DATASET_v2/{optimizer}_time.txt", "w", encoding="utf-8") as file:
        file.write(str(end_time - start_time))

    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("/home/maria/TFM/data/datasets/unfiltered_DATASET/results")
    # results = model("malaga_noche/", save=True)
