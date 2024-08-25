import cv2
import os


def get_image_info(image_path: str):
    height, width, _ = cv2.imread(image_path).shape
    return height, width


if __name__ == "__main__":
    general_dir = r'C:\Users\sierr\Documents\Uni\TFM\archive\for_relabeling'
    unlabeled_images_folder = general_dir + r'\unlabeled_images'

    min_dim = 100000
    min_img = ""
    for img in os.listdir(unlabeled_images_folder):
        if not (img.endswith(".jpg") or img.endswith(".png")):
            continue
        img_path = os.path.join(unlabeled_images_folder, img)
        height, width = get_image_info(img_path)
        if height < min_dim or width < min_dim:
            min_dim = min(height, width)
            min_img = img_path

    print(min_img, ": ", min_dim)
