# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import os
import shutil
import re
from pathlib import Path

GEN_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "reordered_DATASET")
IMAGES_DIR = Path(GEN_DIR, "images")
LABELS_DIR = Path(GEN_DIR, "labels")
DS_INFO_DIR = Path(GEN_DIR, "dataset_info")
NEW_DS_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "filtered_DATASET_v2")
NEW_RAW_DIR = Path(NEW_DS_DIR, "raw_dataset")
NEW_LABELS_DIR = Path(NEW_DS_DIR, "labels")
NEW_DS_INFO_DIR = Path(NEW_DS_DIR, "dataset_info")

FULLFPS = Path(os.environ["ROOT_DIR"], "auxiliar", "FULLFPS")
VA30 = Path(os.environ["ROOT_DIR"], "auxiliar", "VA30")


def copy_useful_to_new_ds(names: list[str], useful_img: set[str], useless_img: set[str]):
    # Check if directories exist, create them if necessary
    NEW_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEW_RAW_DIR.mkdir(parents=True, exist_ok=True)
    NEW_DS_INFO_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        ds_info_path = Path(DS_INFO_DIR, name)
        with open(ds_info_path, "r", encoding="utf-8") as file:
            line = file.readline()
        colors = line.strip().split(" ")

        if name[:-4] in useful_img or ("no" not in colors and (name[:-4] not in useless_img)):

            shutil.copy(Path(IMAGES_DIR, name[:-4] + ".jpg"),
                        Path(NEW_RAW_DIR, name[:-4] + ".jpg"))
            shutil.copy(Path(LABELS_DIR, name), Path(NEW_RAW_DIR, name))
            shutil.copy(Path(DS_INFO_DIR, name), Path(NEW_DS_INFO_DIR, name))


def get_repetitive_images():
    pattern_FULLFPS = re.compile(r"(\d+)_FULLFPS_(\d+)(\D+)\.jpg")
    for f in os.listdir(FULLFPS):
        if f.endswith((".jpg",)) and (mat := pattern_FULLFPS.fullmatch(f)):
            new_dir = Path(FULLFPS)
            new_dir.mkdir(exist_ok=True)
            shutil.copy(Path(IMAGES_DIR, f), Path(new_dir, f))

    pattern_VA30 = re.compile(r"(\d+)_VA30_(\d+)(\D+)\.jpg")
    for f in os.listdir(VA30):
        if f.endswith((".jpg",)) and (mat := pattern_VA30.fullmatch(f)):
            new_dir = Path(VA30)
            new_dir.mkdir(exist_ok=True)
            shutil.copy(Path(IMAGES_DIR, f), Path(new_dir, f))


def save_useful_names():
    for f in os.listdir(FULLFPS):
        with open("all_FULLFPS.txt", "w", encoding='utf-8') as file:
            for f in os.listdir(FULLFPS):
                if f.endswith(".jpg"):
                    file.write(f"{f[:-4]}\n")
        with open("all_VA30.txt", "w", encoding='utf-8') as file:
            for f in os.listdir(VA30):
                if f.endswith(".jpg"):
                    file.write(f"{f[:-4]}\n")
        with open("useful_FULLFPS.txt", "w", encoding='utf-8') as file:
            for f in os.listdir(FULLFPS.joinpath("useful")):
                if f.endswith(".jpg"):
                    file.write(f"{f[:-4]}\n")
        with open("useful_VA30.txt", "w", encoding='utf-8') as file:
            for f in os.listdir(VA30.joinpath("useful")):
                if f.endswith(".jpg"):
                    file.write(f"{f[:-4]}\n")


def get_useful():
    with open("useful_FULLFPS.txt", "r", encoding='utf-8') as file:
        useful_FULLFPS = set([f.strip() for f in file.readlines()])
    with open("useful_VA30.txt", "r", encoding='utf-8') as file:
        useful_VA30 = set([f.strip() for f in file.readlines()])
    with open("all_FULLFPS.txt", "r", encoding='utf-8') as file:
        all_FULLFPS = set([f.strip() for f in file.readlines()])
    with open("all_VA30.txt", "r", encoding='utf-8') as file:
        all_VA30 = set([f.strip() for f in file.readlines()])

    useful_set = useful_FULLFPS | useful_VA30
    all_images = all_FULLFPS | all_VA30
    
    useless_set = all_images - useful_set

    return useful_set, useless_set


if __name__ == "__main__":
    useful, useless = get_useful()
    copy_useful_to_new_ds(os.listdir(DS_INFO_DIR), useful, useless)

