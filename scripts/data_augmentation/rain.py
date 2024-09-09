import albumentations as A
import cv2
import random


def rain(image):
    transform = A.Compose(
        [A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],
    )
    random.seed(7)
    return transform(image=image)


image = cv2.imread('C:\\Users\\sierr\\Documents\\Uni\\TFM\\data\\datasets\\filtered_DATASET\\images\\train\\00002_240405135416_79.jpg')
augmented_image = rain(image)

# Save the result

cv2.imwrite("C:\\Users\\sierr\\Documents\\Uni\\TFM\\pruebas\\rain_00002_240405135416_79.jpg", augmented_image["image"])
