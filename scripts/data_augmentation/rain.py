import albumentations as A
import cv2
import random

def rain(image):
    transform_scale_rot = A.Compose([
       A.CoarseDropout(max_holes=8, max_height=62, max_width=72, p=1)




])
    random.seed(860)
    return transform_scale_rot(image=image)


image = cv2.imread('C:\\Users\\sierr\\Documents\\Uni\\TFM\\data\\datasets\\filtered_DATASET\\images\\train\\00002_240405135416_79.jpg')
augmented_image = rain(image)

# Save the result

cv2.imwrite("aug.jpg", augmented_image["image"])
