import os
import shutil
import random
from pathlib import Path

def split_data(folder_in, folder_out, train_pct):
    train_folder = "train"
    validation_folder = "val"
    img_folder = os.path.join(folder_out, "images")
    label_folder = os.path.join(folder_out, "labels")
    train_img_folder = os.path.join(img_folder, train_folder)
    validation_img_folder = os.path.join(img_folder, validation_folder)
    train_label_folder = os.path.join(label_folder, train_folder)
    validation_label_folder = os.path.join(label_folder, validation_folder)

    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)
    if not os.path.exists(validation_img_folder):
        os.makedirs(validation_img_folder)
    if not os.path.exists(train_label_folder):
        os.makedirs(train_label_folder)
    if not os.path.exists(validation_label_folder):
        os.makedirs(validation_label_folder)

    files = [f for f in os.listdir(folder_in) if f.endswith(".jpg")]
    print(len(files))
    random.shuffle(files)
    train_size = int(len(files) * train_pct / 100)
    train_files = files[:train_size]
    validation_files = files[train_size:]

    for f in train_files:
        img_file = os.path.join(folder_in, f)
        label_file = os.path.join(folder_in, f.replace(".jpg", ".txt"))
        shutil.move(img_file, train_img_folder)
        shutil.move(label_file, train_label_folder)

    for f in validation_files:
        img_file = os.path.join(folder_in, f)
        label_file = os.path.join(folder_in, f.replace(".jpg", ".txt"))
        shutil.move(img_file, validation_img_folder)
        shutil.move(label_file, validation_label_folder)


folder_in = Path(r"Data\Datasets").joinpath("DATASET_v1").joinpath("raw_dataset")
folder_out =Path(r"Data\Datasets").joinpath("DATASET_v1")
train_pct = 90
split_data(folder_in, folder_out, train_pct)
