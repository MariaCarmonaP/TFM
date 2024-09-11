import cv2
import os


def get_image_info(image_path: str):
    height, width, _ = cv2.imread(image_path).shape
    return height, width


if __name__ == "__main__":
    general_dir = r'C:\Users\sierr\Documents\Uni\TFM\data\datasets\filtered_DATASET_v2\raw_dataset'
    unlabeled_images_folder = general_dir 
    max_dim = 0
    min_dim = 100000
    min_img = ""
    max_img = ""
    for img in os.listdir(unlabeled_images_folder):
        if not (img.endswith(".jpg") or img.endswith(".png")):
            continue
        img_path = os.path.join(unlabeled_images_folder, img)
        height, width = get_image_info(img_path)
        if height < min_dim or width < min_dim:
            min_dim = min(height, width)
            min_img = img_path
        if height > max_dim or width > max_dim:
            max_dim = max(height, width)
            max_img = img_path

    print(min_img, ": ", min_dim)
    print(max_img, ": ", max_dim)
