#  this is used by most of the segmentation tasks to find the performance and loss.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

#  iou is basically a intersection of union matrix 
#  this is mainly used in most of the segmentation tasks
smooth = 1e-15
def iou(y_correct_value,y_predicted_value):
    def my_numpy_function(y_correct_value,y_predicted_value):
        intersection = (y_correct_value * y_predicted_value).sum()
        union = y_correct_value.sum() + y_predicted_value.sum() - intersection
        # iou - intersection over union = x
        x_iou = (intersection + smooth) / (union + smooth)
        
        # More Info: https://www.askpython.com/python/built-in-methods/python-astype#:~:text=Python%20astype()%20method%20enables,form%20using%20astype()%20function.
        # As type function is used to change the datatype to another data column
        # Enables us to set or convert the data type of an existing data column in a dataset or a data frame
        x_iou = x_iou.astype(np.float32)
        return x_iou
    # More info: https://www.tensorflow.org/api_docs/python/tf/numpy_function
    # The the sata inside the square brackets are the [input] 
    # / here the input is the [y_correct_value and the y_predicted_value]
    return tf.numpy_function(my_numpy_function,[y_correct_value,y_predicted_value],tf.float32)


#  this funtions is the dice coefficent 
#  it is much similar to the iou and is used to calculate the performance of rhe segmenetation task
#  More Info: https://www.kaggle.com/code/yerramvarun/understanding-dice-coefficient
def dice_coeff(y_correct_value,y_predicted_value):
    # Flattens the input. Does not affect the batch size.
    # More Info: https://keras.io/api/layers/reshaping_layers/flatten/
    y_correct_value = tf.keras.layers.Flatten()(y_correct_value)
    y_predicted_value = tf.keras.layers.Flatten()(y_predicted_value)

    # reduce sumis used to change the dimension of the array to an smaller one\
    # More Info: https://stackoverflow.com/questions/47157692/how-does-reduce-sum-work-in-tensorflow
    # More Info: https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum
    intersection = tf.reduce_sum(y_correct_value * y_predicted_value)
    return (2.*intersection+smooth) / (tf.reduce_sum(y_correct_value) * tf.reduce_sum(y_predicted_value)+smooth)

#  this function is "1 - dice coefficient" 
def dice_loss(y_correct_value,y_predicted_value):
    return 1.0 - dice_coeff(y_correct_value,y_predicted_value)

