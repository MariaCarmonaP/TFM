import albumentations as A
import cv2
import os
import json
import random


# Helper function to read YOLO labels
def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        labels = []
        for line in file:
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append([x_center, y_center, width, height, cls])
    return labels

# Helper function to write YOLO labels
def write_yolo_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            cls = int(label[-1])
            x_center, y_center, width, height = label[:-1]
            file.write(f"{cls} {x_center} {y_center} {width} {height}\n")

# Helper function to read dataset info
def read_dataset_info(info_path):
    with open(info_path, 'r') as file:
        info = []
        lines = file.readlines()
        colors = lines[0].strip().split()
        fronts = [elem for elem in lines[1].strip().split()]
        dists = [elem for elem in lines[2].strip().split()]
        for index, color in enumerate(colors):
            info.append([color, fronts[index], dists[index]])
    return info

# Helper function to write dataset info
def write_dataset_info(info_path, info):
    colors = " ".join([info_vehicle[0] for info_vehicle in info])
    fronts = " ".join([info_vehicle[1] for info_vehicle in info])
    dists = " ".join([info_vehicle[2] for info_vehicle in info])

    with open(info_path, 'w') as file:
        file.write(colors + "\n")
        file.write(fronts + "\n")
        file.write(dists + "\n")

transform_scale_rot = A.Compose(
    [
        A.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit_y=0.005, shift_limit_x=0.3, p=1),
        A.RandomResizedCrop(height=608, width=608, scale=(0.6, 0.9), ratio=(0.99, 1.01), p=1.0),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, p=0.2),
        A.RandomRain(
            slant_lower=-5,
            slant_upper=5,
            drop_length=10,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=1,
            brightness_coefficient=0.9,
            p=0.1,
        ),
        A.RandomShadow(shadow_roi=(0.5, 0.5, 1.0, 1.0), p=0.1),
        A.CoarseDropout(max_holes=2, max_height=32, max_width=32, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids', "dataset_info"])
)



transform_color = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=5, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.4, p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=0.8
        ),
        A.RandomRain(rain_type="drizzle", p=0.3),
        A.RandomSunFlare(p=0.1),
        A.RandomShadow(shadow_roi=(0.5, 0.5, 1.0, 1.0), p=0.1),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids', "dataset_info"])
)
transform_rain = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=15,
            drop_width=2,
            drop_color=(170, 170, 170),
            blur_value=3,
            brightness_coefficient=0.8,
            p=0.7,
        )], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids', "dataset_info"]))

transform_weather = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0, rotate_limit=10, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, p=0.3),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.2),
        A.RandomSunFlare(p=0.3),
        A.RandomShadow(
            shadow_roi=(0.5, 0.5, 1.0, 1.0),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=3,
            p=0.4,
        ),
        A.CoarseDropout(max_holes=3, max_height=42, max_width=32, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids', "dataset_info"])
)

transform_dropout = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=5, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, p=0.3),
        A.RandomRain(
            slant_lower=-15,
            slant_upper=15,
            drop_length=20,
            drop_width=3,
            drop_color=(150, 150, 150),
            blur_value=5,
            brightness_coefficient=0.7,
            p=0.2,
        ),
        A.RandomSunFlare(p=0.1),
        A.RandomShadow(shadow_roi=(0.5, 0.5, 1.0, 1.0), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=42, max_width=32, p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids', "dataset_info"])
)

FILTERED_IMAGES_DIR = r"C:\Users\sierr\Documents\Uni\TFM\pruebas\pruebas_augment\images"
FILTERED_LABELS_DIR = r"C:\Users\sierr\Documents\Uni\TFM\pruebas\pruebas_augment\labels"
FILTERED_INFO_DIR = r"C:\Users\sierr\Documents\Uni\TFM\pruebas\pruebas_augment\dataset_info"
SCRIPTS_DIR = r"C:\Users\sierr\Documents\Uni\TFM\scripts"
# set seed for random generated numbers
random.seed(64330)

# load json data

with open(
    os.path.join(SCRIPTS_DIR, "augment_prob.json"),
    "r",
    encoding="utf-8",
) as f:
    augment_probs = json.load(f)

with open(
    os.path.join(SCRIPTS_DIR, "similar_images.json"),
    "r",
    encoding="utf-8",
) as f:
    similar_images = json.load(f)

for file_name in os.listdir(FILTERED_IMAGES_DIR):
    if file_name.split("_")[1] in ["augScaleRot", "augColor", "augWeather", "augDropout", "augRain"]:
        continue
    if file_name.endswith(".jpg"):
        image = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, file_name))
        label_path = os.path.join(FILTERED_LABELS_DIR, file_name.replace(".jpg", ".txt"))
        labels = read_yolo_labels(label_path)
        info_path = os.path.join(FILTERED_INFO_DIR, file_name.replace(".jpg", ".txt"))
        info = read_dataset_info(info_path)
        # Separate bounding boxes and category ids
        bboxes = [label[:4] for label in labels]
        category_ids = [label[4] for label in labels]

        if random.random() <= augment_probs[file_name[:-4]] or True:
            if file_name in similar_images:
                to_augment = similar_images[file_name].pop(random.randrange(len(similar_images[file_name])))
            else:
                to_augment = file_name
            image_to_augment = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, to_augment))
            

            augmented = transform_scale_rot(image=image_to_augment, bboxes=bboxes, category_ids=category_ids, dataset_info=info)
            new_name = file_name[:5] + "_augScaleRot_" + file_name[5:]
            cv2.imwrite(os.path.join(FILTERED_IMAGES_DIR, new_name), augmented["image"])
            new_label_name = new_name.replace(".jpg", ".txt")
            new_label_path = os.path.join(FILTERED_LABELS_DIR, new_label_name)
            write_yolo_labels(new_label_path, [list(bbox) + [category_id] for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])])
            info_path = os.path.join(FILTERED_INFO_DIR, new_label_name)
            write_dataset_info(info_path, augmented["dataset_info"])

        if random.random() <= augment_probs[file_name[:-4]] or True:
            if file_name in similar_images:
                to_augment = similar_images[file_name].pop(random.randrange(len(similar_images[file_name])))
            else:
                to_augment = file_name
            image_to_augment = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, to_augment))
            augmented = transform_color(image=image_to_augment, bboxes=bboxes, category_ids=category_ids, dataset_info=info)
            new_name = file_name[:5] + "_augColor_" + file_name[5:]
            cv2.imwrite(os.path.join(FILTERED_IMAGES_DIR, new_name), augmented["image"])
            new_label_name = new_name.replace(".jpg", ".txt")
            new_label_path = os.path.join(FILTERED_LABELS_DIR, new_label_name)
            write_yolo_labels(new_label_path, [list(bbox) + [category_id] for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])])
            info_path = os.path.join(FILTERED_INFO_DIR, new_label_name)
            write_dataset_info(info_path, augmented["dataset_info"])
        
        if random.random() <= augment_probs[file_name[:-4]] or True:
            if file_name in similar_images:
                to_augment = similar_images[file_name].pop(random.randrange(len(similar_images[file_name])))
            else:
                to_augment = file_name
            image_to_augment = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, to_augment))
            augmented = transform_rain(image=image_to_augment, bboxes=bboxes, category_ids=category_ids, dataset_info=info)
            new_name = file_name[:5] + "_augRain_" + file_name[5:]
            cv2.imwrite(os.path.join(FILTERED_IMAGES_DIR, new_name), augmented["image"])
            new_label_name = new_name.replace(".jpg", ".txt")
            new_label_path = os.path.join(FILTERED_LABELS_DIR, new_label_name)
            write_yolo_labels(new_label_path, [list(bbox) + [category_id] for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])])
            info_path = os.path.join(FILTERED_INFO_DIR, new_label_name)
            write_dataset_info(info_path, augmented["dataset_info"])

        if random.random() <= augment_probs[file_name[:-4]] or True:
            if file_name in similar_images:
                to_augment = similar_images[file_name].pop(random.randrange(len(similar_images[file_name])))
            else:
                to_augment = file_name
            image_to_augment = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, to_augment))
            augmented = transform_weather(image=image_to_augment, bboxes=bboxes, category_ids=category_ids, dataset_info=info)
            new_name = file_name[:5] + "_augWeather_" + file_name[5:]
            cv2.imwrite(os.path.join(FILTERED_IMAGES_DIR, new_name), augmented["image"])
            new_label_name = new_name.replace(".jpg", ".txt")
            new_label_path = os.path.join(FILTERED_LABELS_DIR, new_label_name)
            write_yolo_labels(new_label_path, [list(bbox) + [category_id] for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])])
            info_path = os.path.join(FILTERED_INFO_DIR, new_label_name)
            write_dataset_info(info_path, augmented["dataset_info"])
        
        if random.random() <= augment_probs[file_name[:-4]] or True:
            if file_name in similar_images:
                to_augment = similar_images[file_name].pop(random.randrange(len(similar_images[file_name])))
            else:
                to_augment = file_name
            image_to_augment = cv2.imread(os.path.join(FILTERED_IMAGES_DIR, to_augment))
            augmented = transform_dropout(image=image_to_augment, bboxes=bboxes, category_ids=category_ids, dataset_info=info)
            new_name = file_name[:5] + "_augDropout_" + file_name[5:]
            cv2.imwrite(os.path.join(FILTERED_IMAGES_DIR, new_name), augmented["image"])
            new_label_name = new_name.replace(".jpg", ".txt")
            new_label_path = os.path.join(FILTERED_LABELS_DIR, new_label_name)
            write_yolo_labels(new_label_path, [list(bbox) + [category_id] for bbox, category_id in zip(augmented['bboxes'], augmented['category_ids'])])
            info_path = os.path.join(FILTERED_INFO_DIR, new_label_name)
            write_dataset_info(info_path, augmented["dataset_info"])

    # save the updated similar images json
    with open(
        os.path.join(SCRIPTS_DIR, "similar_images_after_augmenting.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(similar_images, f, indent=4)