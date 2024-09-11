# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import os
import shutil
from pathlib import Path
import json


# GEN_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "unfiltered_DATASET")
# PATHS 
GEN_DIR = Path(os.environ["ROOT_DIR"], "archive", "for_relabelling")
IMAGES_DIR = Path(GEN_DIR, "unlabeled_images")
LABELS_DIR = Path(GEN_DIR, "new_labels")
DS_INFO_DIR = Path(GEN_DIR, "dataset_info")
NEW_DS_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "reordered_DATASET")
NEW_IMAGES_DIR = Path(NEW_DS_DIR, "images")
NEW_LABELS_DIR = Path(NEW_DS_DIR, "labels")
NEW_DS_INFO_DIR = Path(NEW_DS_DIR, "dataset_info")

# CONSTANTS
TOTAL_VEHICLES = (86, 1865, 1418, 1853, 130, 412, 366, 3)


def reorder(file_names: list[str]) -> None:
    for i, name in enumerate(file_names):
        new_name = f"{(i+1):05d}{"_"+name}"
        if not (og_img_path := Path(IMAGES_DIR, name+".jpg")).exists():
            print(f"File {name}.jpg not found")
            continue
        if not (og_label_path := Path(LABELS_DIR, name+".txt")).exists():
            print(f"File {name}.txt not found")
            continue
        if not (og_ds_info_path := Path(DS_INFO_DIR, name+".txt")).exists():
            print(f"Dataset info file {name}.txt not found")
            continue
        shutil.copy(og_img_path, Path(NEW_IMAGES_DIR, new_name+".jpg"))
        shutil.copy(og_label_path, Path(NEW_LABELS_DIR, new_name+".txt"))
        shutil.copy(og_ds_info_path, Path(NEW_DS_INFO_DIR, new_name+".txt"))


def save_camera_info(file_names: list[str]) -> None:
    camera: dict[str, str] = {}
    for name in file_names:
        for dire in [Path(NEW_DS_DIR, f"cam{i:02d}") for i in range(1, 12)]:
            if name+".jpg" in os.listdir(dire):
                camera[name] = dire.name
                break

    with open(Path("scripts", "camera.json"), "w", encoding="utf-8") as f:
        json.dump(camera, f, indent=4)


def save_augment_probs(file_names: list[str]) -> None:
    augment_prob: dict[str, float] = {}

    n_ideal = max(TOTAL_VEHICLES)
    adjust_prob = min(TOTAL_VEHICLES)/n_ideal
    class_augment_coef = tuple((n_ideal - total_class)/n_ideal for total_class in TOTAL_VEHICLES)

    for name in file_names:

        with open(Path(NEW_DS_INFO_DIR, name+".txt"), "r", encoding="utf-8") as f:
            distances = [int(d) for d in f.readlines()[2].strip().split(" ")]
            dist = "far"
            for distance in distances:
                if distance < 2:
                    dist = "close"

        with open(Path(NEW_LABELS_DIR, name+".txt"), "r", encoding="utf-8") as f:
            classes = [int(line.split(" ")[0]) for line in f.readlines()]

        prob = 0.0
        for cl in classes:
            prob += class_augment_coef[cl]
        augment_prob[name] = (prob/len(classes) + adjust_prob)*(0.6 if dist == "close" else 0.3)

    with open(Path("scripts", "augment_prob.json"), "w", encoding="utf-8") as f:
        json.dump(augment_prob, f, indent=4)


if __name__ == "__main__":
    names = [f[:-4] for f in os.listdir(NEW_IMAGES_DIR) if f.endswith((".jpg"))]
    save_augment_probs(names)
