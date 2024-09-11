# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

import json
from pathlib import Path


FILTERED_DIR = (
    r"C:\Users\sierr\Documents\Uni\TFM\data\datasets\filtered_DATASET_v2\raw_dataset"
)
UNFILTERED_DIR = (
    r"C:\Users\sierr\Documents\Uni\TFM\data\datasets\reordered_DATASET\images"
)
prev_file = ""
similar_dict: dict[str, list[str]] = {}

filtered_images = [p.stem for p in sorted(Path(FILTERED_DIR).rglob("*.jpg"))]
for file_path in sorted(Path(UNFILTERED_DIR).rglob("*.jpg")):
    file_name = file_path.stem
    if file_name in filtered_images:
        similar_dict[file_name] = []
        prev_file = file_name
        continue
    if prev_file in similar_dict:
        similar_dict[prev_file].append(file_name)
    else:
        similar_dict[prev_file] = [file_name]

to_delete = []
for file, similar_files in similar_dict.items():
    if not similar_files:
        to_delete.append(file)
for file in to_delete:
    del similar_dict[file]


with open("similar_images.json", "w", encoding="utf-8") as f:
    json.dump(similar_dict, f, indent=4)
