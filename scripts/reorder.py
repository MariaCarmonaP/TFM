import os
import shutil
from pathlib import Path

#GEN_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "unfiltered_DATASET")
GEN_DIR = Path(os.environ["ROOT_DIR"], "archive", "for_relabelling")
IMAGES_DIR = Path(GEN_DIR, "unlabeled_images")
LABELS_DIR = Path(GEN_DIR, "new_labels")
DS_INFO_DIR = Path(GEN_DIR, "dataset_info")
NEW_DS_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "reordered_DATASET")
NEW_IMAGES_DIR = Path(NEW_DS_DIR, "images")
NEW_LABELS_DIR = Path(NEW_DS_DIR, "labels")
NEW_DS_INFO_DIR = Path(NEW_DS_DIR, "dataset_info")

if __name__ == "__main__":
    file_names = [f[:-4] for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg"))]
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
