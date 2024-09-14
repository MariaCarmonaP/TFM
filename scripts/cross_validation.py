import os
import yaml
import json
import pandas as pd
import datetime
import shutil
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold
from ultralytics import YOLO
from torch import cuda

dataset_path = Path(
    os.environ["ROOT_DIR"], r"data/datasets/filtered_DATASET_v2"
)
labels = sorted((dataset_path / "labels").rglob("*.txt"))
yaml_file = Path(
    os.environ["ROOT_DIR"], r"data/datasets/filtered_DATASET_v2/cfg.yaml"
)  # your data YAML with data directories and names dictionary
with open(yaml_file, "r", encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

indx = [label.stem for label in labels]  # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

for label in labels:
    lbl_counter = Counter()

    with open(label, "r", encoding="utf-8") as lf:
        lines = lf.readlines()

    for line in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(line.split(" ")[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

ksplit = 4

kf = KFold(n_splits=ksplit, shuffle=True, random_state=444)  # setting random_state for repeatable results

kfolds = list(kf.split(labels_df))

folds = [f"split_{n}" for n in range(1, ksplit + 1)]
folds_df = pd.DataFrame(index=indx, columns=folds)

for idx, (train, val) in enumerate(kfolds, start=1):
    folds_df[f"split_{idx}"].loc[labels_df.iloc[train].index] = "train"
    folds_df[f"split_{idx}"].loc[labels_df.iloc[val].index] = "val"

fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1e-7)
    fold_lbl_distrb.loc[f"split_{n}"] = ratio


supported_extensions = [".jpg", ".jpeg", ".png"]

# Initialize an empty list to store image file paths
images = []

# Loop through supported extensions and gather image files
for ext in supported_extensions:
    images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

# Create the necessary directories and dataset YAML files (unchanged)
save_path = Path(dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
save_path.mkdir(parents=True, exist_ok=True)
ds_yamls = []

for split in folds_df.columns:
    # Create directories
    split_dir = save_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create dataset YAML files
    dataset_yaml = split_dir / f"{split}_dataset.yaml"
    ds_yamls.append(dataset_yaml)

    with open(dataset_yaml, "w", encoding="utf-8") as ds_y:
        yaml.safe_dump(
            {
                "path": split_dir.as_posix(),
                "train": "train",
                "val": "val",
                "names": classes,
            },
            ds_y,
        )



for image, label in zip(images, labels):
    for split, k_split in folds_df.loc[image.stem].items():
        # Destination directory
        img_to_path = save_path / split / k_split / "images"
        lbl_to_path = save_path / split / k_split / "labels"

        # Copy image and label files to new directory (SamefileError if file already exists)
        shutil.copy(image, img_to_path / image.name)
        shutil.copy(label, lbl_to_path / label.name)

folds_df.to_csv(save_path / "kfold_datasplit.csv")
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")


weights_path = os.environ["ROOT_DIR"] + os.sep + "data/results/filtered_DATASET_v2/PSO_batch/14_23_16_1e-06_0.4973647142714877/weights/best.pt"

model = YOLO("yolov8n.pt", task="detect")


results = {}

# Define your additional arguments here
batch = 16
epochs = 40
weight_decay = 1e-06
cls= 0.4973647142714877

for k in range(ksplit):
    dataset_yaml = ds_yamls[k]
    model.train(
        data=dataset_yaml,
        project=r"/home/maria/TFM/data/results/filtered_DATASET_v2/kfold_5",
        name=f"{k}_fold",
        epochs=40,
        imgsz=608,
        device="cuda:0" if cuda.is_available() else "cpu",
        exist_ok=True,
        seed=443,
        optimizer="Adam",
        # close_mosaic=0,
        # hsv_h=0,
        # hsv_s=0,
        # hsv_v=0,
        # translate=0,
        # scale=0,
        # fliplr=0,
        # mosaic=0,
        cos_lr=True,
        batch=batch,
        lr0=0.0010192339694602597,
        lrf=0.01,
        momentum=0.9171775316059347,
        weight_decay=weight_decay,
        cls=cls,
    )
    results[k] = model.val()  # save output metrics for further analysis
    try:
        # Save output metrics to a file
        with open(f"/home/maria/TFM/data/results/filtered_DATASET_v2/kfold_5/output_metrics_{k}.json", "w") as f:
            json.dump(results[k].results_dict, f)
            
    except Exception:
        pass