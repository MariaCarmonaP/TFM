# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import os

LABEL_FOLDER = r"/home/maria/TFM/data/datasets/filtered_DATASET_v2/labels"


def read_yolo_labels(label_path):
    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    classes = []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        classes.append(class_id)

    return classes


def count_classes(LABEL_FOLDER):
    n_clases = {clase: 0 for clase in range(8)}
    files = [f for f in os.listdir(LABEL_FOLDER) if f.endswith((".txt",))]
    for f in files:
        classes = read_yolo_labels(os.path.join(LABEL_FOLDER, f))
        for clase in classes:
            n_clases[clase] += 1
    return n_clases


if __name__ == "__main__":

    distribucion_clases = count_classes(LABEL_FOLDER)
    print("M: ", distribucion_clases[0])
    print("C: ", distribucion_clases[1])
    print("FL: ", distribucion_clases[2])
    print("FP: ", distribucion_clases[3])
    print("A: ", distribucion_clases[4])
    print("CL: ", distribucion_clases[5])
    print("CP: ", distribucion_clases[6])
    print("CPA: ", distribucion_clases[7])
