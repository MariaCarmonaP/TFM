import os
import cv2

class Box:
    def __init__(self, center_x, center_y, height, width):
        self.center_x = center_x
        self.center_y = center_y
        self.height = height
        self.width = width

    def get_left_x(self):
        return self.center_x - self.width / 2

    def get_right_x(self):
        return self.center_x + self.width / 2

    def get_top_y(self):
        return self.center_y - self.height / 2

    def get_bottom_y(self):
        return self.center_y + self.height / 2

    def overlaps(self, other_box):
        left_x1, right_x1 = self.get_left_x(), self.get_right_x()
        top_y1, bottom_y1 = self.get_top_y(), self.get_bottom_y()
        left_x2, right_x2 = other_box.get_left_x(), other_box.get_right_x()
        top_y2, bottom_y2 = other_box.get_top_y(), other_box.get_bottom_y()

        # Check if boxes overlap
        if (left_x1 <= right_x2 and left_x2 <= right_x1 and
            top_y1 <= bottom_y2 and top_y2 <= bottom_y1):
            return True
        return False
    
def balance_classes(all_classes: list[int]):
    total_img = sum(all_classes)
