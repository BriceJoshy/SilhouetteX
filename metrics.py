#  this is used by most of the segmentation tasks to find the performance and loss.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

#  iou is basically a intersection of union matrix 
#  this is mainly used in most of the segmentation tasks
def iou(y_correct_value,y_predicted_value):
    def f(y_correct_value,y_predicted_value):
        intersection = (y_correct_value * y_predicted_value).sum()
        union = y_correct_value.sum() + y_predicted_value.sum() - intersection
        # iou - intersection over union = x
        x_iou = (intersection + 1e-15) / (union + 1e-15)
