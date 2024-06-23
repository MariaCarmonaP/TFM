import json

# Specify the class list
classes = ['M','C','FL','FP','A','CL','CP','CPA']

# Load the json file
with open('C:\\Users\\sierr\\Documents\\Uni\\TFM\\jsonmin.json', 'r') as f:
    data = json.load(f)

# Loop through each image in the data
for image in data:
    # Create a txt file with the same name as the image
    path = 'C:\\Users\\sierr\\Documents\\Uni\\TFM\\FURGONETAS_LIGERAS_Y_PESADAS_LABELS\\'
    filename = image['image'].split('/')[-1].split('.')[0].split('-')[1] + '.txt'
    
    with open(path + filename, 'w') as file:
        # Loop through each label in the image
        print(image['id'])
        if 'label' in image:
            for label in image['label']:
                WIDTH = int(label['original_width'])
                HEIGHT = int(label['original_height'])
                # Get the class index
                class_index = classes.index(label['rectanglelabels'][0])
                x=float(label['x'])/100 * WIDTH
                y=float(label['y'])/100 * HEIGHT
                w=float(label['width'])/100 * WIDTH
                h=float(label['height'])/100 * HEIGHT

                x = (x +w/2)/WIDTH
                w = (w)/WIDTH
                y = (y +h/2)/HEIGHT
                h = (h)/HEIGHT

                print(class_index,x,y,w,h)
                # Write the coordinates in the yolov5 format, scaling the percentages to be between 0 and 1
                file.write(f'{class_index} {x} {y} {w} {h}\n')
                #     class_index=class_index,
                #     x=label['x'] / 100,
                #     y=label['y'] / 100,
                #     width=label['width'] / 100,
                #     height=label['height'] / 100
                # ))
