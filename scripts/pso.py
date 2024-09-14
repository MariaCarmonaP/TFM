import random
from ultralytics import YOLO  # type: ignore
from torch import cuda

w = 0.7  # constant inertia weight (how much to weigh the previous velocity)
c1 = 1.7  # cognative constant
c2 = 1.7  # social constant
