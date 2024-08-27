
import cv2
import os
import numpy as np
from skimage import io, util

# Load existing YOLO labels and images
labels_dir = 'path/to/labels'
images_dir = 'path/to/images'
labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
images_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

class_distribution = {}
for label_file in labels_files:
    with open(os.path.join(labels_dir, label_file), 'r') as f:
        for line in f:
            class_id, _, _, _, _ = line.strip().split()
            class_id = int(class_id)
            if class_id not in class_distribution:
                class_distribution[class_id] = 0
            class_distribution[class_id] += 1

# Identify underrepresented classes
min_instances = min(class_distribution.values())
underrepresented_classes = [class_id for class_id, count in class_distribution.items() if count < min_instances]

# Generate cutouts for underrepresented classes
for image_file in images_files:
    image_path = os.path.join(images_dir, image_file)
    img = io.imread(image_path)
    for class_id in underrepresented_classes:
        # Load corresponding labels for this image
        labels = []
        with open(os.path.join(labels_dir, image_file.replace('.jpg', '.txt')), 'r') as f:
            for line in f:
                class_id_label, _, _, _, _ = line.strip().split()
                class_id_label = int(class_id_label)
                if class_id_label == class_id:
                    x, y, w, h = map(int, line.strip().split()[1:])
                    labels.append((x, y, w, h))

        # Iterate over bounding boxes for this class
        for x, y, w, h in labels:
            # Extract non-overlapping ROI (Region of Interest)
            roi = img[y:y+h, x:x+w]
            # Save ROI as a new image
            roi_path = f'roi_{image_file}_{class_id}.jpg'
            io.imsave(roi_path, roi)

print(f"Generated {len(underrepresented_classes)} underrepresented class ROI images")