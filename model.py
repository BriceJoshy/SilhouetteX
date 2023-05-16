# The network structure combining Deeplabv3 and ResNet 50(Residual network)
# Building the deeplab_v3 of human segementation
# More Info : https://learnopencv.com/deeplabv3-ultimate-guide/
# More Info: https://idiotdeveloper.com/deeplabv3-resnet50-architecture-in-tensorflow-using-keras/
#  using the resnet50 as the encoder

import os
# setting the value for the environment variable TF_CPP_MIN_LOG_LEVEL to 2. 
# This prevents showing information/warning logs from TensorFlow.
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
    #  this variable passes to the entire network 
    """ Input For the layers """
    inputs = Input(shape)

    """Encoder"""
    #  importing the encoder
    #  include top basically means that if any classifier should be added at the end  of the network
    #  using the encoder for the segmentation task so there is not classification work is done i.e why the include top is set to False
    #  tensor input is the input definined above , i.e the input for the entire network
    #  the input passes to the resnet encoder
    #  first the input passes through the resnet50 encoder then we take the output i.e the image features
    encoder  = ResNet50(weights="imagenet",include_top=False,input_tensor=inputs)

    # More Info about conv4_block6_out: https://stackoverflow.com/questions/60234176/unpack-tf-keras-model-layer
    #  we are setting up the parameters of the image as conv4_block_out
    image_features = encoder.get_layer("conv4_block6_out")

