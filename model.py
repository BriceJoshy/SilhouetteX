# The network structure combining Deeplabv3 and ResNet 50
# Building the deeplab_v3 of human segementation
# More Info : https://learnopencv.com/deeplabv3-ultimate-guide/
# More Info: https://idiotdeveloper.com/deeplabv3-resnet50-architecture-in-tensorflow-using-keras/
#  using the resnet50 as the encoder

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose
from keras.layers import AveragePooling2D, GlobalAveragePooling2D,UpSampling2D,Reshape
from keras.applications import ResNet50
import tensorflow as tf

"""Function for Deeplabv3+"""
#  taking the shape as input 
def deeplabv3_plus(shape):
    # defining the input for the model
    """ Input For the layers """
    inputs = Input(shape)

    """Encoder"""
    #  importing the encoder
    #  include top basically means that if any classifier should be added at the end 
    #  using the encoder for the segmentation task
    encoder  = ResNet50(weights="imagenet",include_top=False)


