import os
import cv2
import numpy as np


def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    classes = []
    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        classes.append(class_id)
    
    return classes


def count_classes(label_folder):
    n_clases = {clase:0 for clase in range(8)}
    files = [f for f in os.listdir(label_folder) if f.endswith(('.txt',))]
    for f in files:
        classes = read_yolo_labels(os.path.join(label_folder, f))
        for clase in classes:
            n_clases[clase] += 1
    return n_clases
        

if __name__ == "__main__":
    label_folder = r"C:\Users\sierr\Documents\Uni\TFM\archive\for_relabelling\new_labels"
    
    n_clases = count_classes(label_folder)
    print('M: ', n_clases[0])
    print('C: ', n_clases[1])
    print('FL: ', n_clases[2])
    print('FP: ', n_clases[3])
    print('A: ', n_clases[4])
    print('CL: ', n_clases[5])
    print('CP: ', n_clases[6])
    print('CPA: ', n_clases[7])