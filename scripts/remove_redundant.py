# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import pylab as p

GEN_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "reordered_DATASET")
IMAGES_DIR = Path(GEN_DIR, "images")
LABELS_DIR = Path(GEN_DIR, "labels")
DS_INFO_DIR = Path(GEN_DIR, "dataset_info")
NEW_DS_DIR = Path(os.environ["ROOT_DIR"], "data", "datasets", "filtered_DATASET")
NEW_IMAGES_DIR = Path(NEW_DS_DIR, "images")
NEW_LABELS_DIR = Path(NEW_DS_DIR, "labels")
NEW_DS_INFO_DIR = Path(NEW_DS_DIR, "dataset_info")


def copy_useful_to_new_ds(names: list[str]):
    for name in names:
        ds_info_path = Path(DS_INFO_DIR, name)
        with open(ds_info_path, "r", encoding="utf-8") as file:
            line = file.readline()
        colors = line.strip().split(" ")
        if "no" not in colors:
            shutil.copy(Path(IMAGES_DIR, name[:-4] + ".jpg"),
                        Path(NEW_IMAGES_DIR, name[:-4] + ".jpg"))
            shutil.copy(Path(LABELS_DIR, name), Path(NEW_LABELS_DIR, name))
            shutil.copy(Path(DS_INFO_DIR, name), Path(NEW_DS_INFO_DIR, name))


def parse_ds_info(names: list[str],
                  ds_info_dir: str,
                  labels_dir: str,
                  ) -> tuple[list[int], dict[str, int], list[int], int, int, int, int]:
    all_classes: list[int] = [0]*8
    all_colors: dict[str, int] = {}
    all_dists: list[int] = [0]*4
    n_undefined_dist = 0
    n_fronts = 0
    n_backs = 0
    n_not_van = 0

    for name in names:
        # get stats
        with open(Path(ds_info_dir, name), "r", encoding="utf-8") as file:
            lines = file.readlines()

        # get classes
        with open(Path(labels_dir, name), "r", encoding="utf-8") as file:
            classes = [int(line.strip().split(" ")[0]) for line in file.readlines()]

        colors = lines[0].strip().split(" ")
        front_back = [int(dist) for dist in lines[1].strip().split(" ")]
        distance = [int(dist) for dist in lines[2].strip().split(" ")]

        for index, color in enumerate(colors):
            clas = classes[index]
            if clas < 0 or clas > 7:
                print("Class error: ", name)
            else:
                all_classes[clas] += 1

            if clas in [2, 3]:
                if color in all_colors:
                    all_colors[color] += 1
                else:
                    all_colors[color] = 1

                f_b = front_back[index]
                if f_b == 1:
                    n_fronts += 1
                elif f_b == 1:
                    n_backs += 1
                else:
                    n_not_van += 1

                d = distance[index]
                if d < 0:
                    n_undefined_dist += 1
                else:
                    all_dists[d] += 1

    return all_classes, all_colors, all_dists, n_undefined_dist, n_fronts, n_backs, n_not_van


def draw_classes(all_classes: list[int]):
    labels = ['M', 'C', 'FL', 'FP', 'A', 'CL', 'CP', 'CPA']

    ax = plt.subplots()[1]
    ax.pie(all_classes, labels=labels)


def draw_colors(all_colors: dict[str, int]):
    _, ax = plt.subplots()
    ax.pie(list(all_colors.values()), labels=list(all_colors.keys()))


def draw_graphs(
               all_dists: list[int],
               n_undefined_dist: int, 
               n_fronts: int,
               n_backs: int,
               n_not_van: int,):
    pass


if __name__ == "__main__":
    file_names = [f for f in os.listdir(NEW_LABELS_DIR) if f.endswith((".txt",))]
    classes, colors, dists, _, _, _, _ = parse_ds_info(file_names, str(NEW_DS_INFO_DIR), str(NEW_LABELS_DIR))
    draw_colors(colors)
    p.show()
    #copy_useful_to_new_ds(file_names)