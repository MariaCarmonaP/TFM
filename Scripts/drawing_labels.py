import os
import cv2
import numpy as np


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0),
          line_type=1
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, line_type)

    return text_size


def read_yolo_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    bounding_boxes = []
    i = 0

    for line in lines:
        values = line.strip().split()
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:])
        
        bounding_boxes.append({
            'class_id': class_id,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'n': i
        })

        i = i + 1
    
    return bounding_boxes

def draw_bounding_boxes(image_path, label_path, output_folder):

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    bounding_boxes = read_yolo_labels(label_path)
    
    for box in bounding_boxes:
        x_center, y_center, box_width, box_height = box['x_center'] * width, box['y_center'] * height, box['width'] * width, box['height'] * height
        x_min, y_min, x_max, y_max = int(x_center - box_width / 2), int(y_center - box_height / 2), int(x_center + box_width / 2), int(y_center + box_height / 2)
        
        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        
        # Optionally, you can also put text with class information
        class_id = box['class_id']
        draw_text(image, f'''{box['n']}''',
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(x_min, y_min - 20),
          font_scale=2,
          font_thickness=3,
          text_color=(0,0,0),
          text_color_bg=(255, 255, 255),
          line_type=cv2.LINE_AA
          )
    
    # Save the image with bounding boxes
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

def process_dataset(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)
        
        draw_bounding_boxes(image_path, label_path, output_folder)

if __name__ == "__main__":
    image_folder = r"C:\Users\sierr\Documents\Uni\TFM\archive\for_relabelling\missed_pics"
    label_folder = r"C:\Users\sierr\Documents\Uni\TFM\archive\for_relabelling\missed_labels"
    output_folder = r"C:\Users\sierr\Documents\Uni\TFM\archive\for_relabelling\bigger_class_rect"
    
    process_dataset(image_folder, label_folder, output_folder)
