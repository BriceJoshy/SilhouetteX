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
from keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
)
from keras.layers import (
    AveragePooling2D,
    GlobalAveragePooling2D,
    UpSampling2D,
    Reshape,
    Input,
)
from keras.applications import ResNet50
import tensorflow as tf

"""Function for Deeplabv3+"""


#  Atrous Spacial Pyramid Pooling for deeplabv3+
#  More Info: https://developers.arcgis.com/python/guide/how-deeplabv3-works/
def ASPP(inputs):
    # Average pooling is  downsizing the image wrt the dimensions of it
    # Avreage pooling is done for the input
    shape = inputs.shape
    #  a pool size of about 32 by 32 as we print the shape we got (none,32,32,1024)
    # print(inputs.shape)
    #  now the feature map is converted to 1:1 by 1024
    # setting the input as 'inputs'  
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    #  now the featuer map is (none,1,1,1024)
    # print(y1.shape)


    #  now we will upsample and convolution layer
    #  256 as features andsize as 1 padding is same as we dont want to change the shape 
    # More Info about convo2d: https://keras.io/api/layers/convolution_layers/convolution2d/ 
    # use_bais is a boolean used to see if the layer used a bias vector
    # More Info about: https://deepai.org/machine-learning-glossary-and-terms/bias-vector
    y1 = Conv2D(256,1,padding="same",use_bias=False)

    """we do the batch_normalization where y1 as input:""" 
    # Layer that normalizes its inputs.
    #  Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1. 
    # More info: https://keras.io/api/layers/normalization_layers/batch_normalization/
    y1 = BatchNormalization()(y1)



#  taking the shape as input
def deeplabv3_plus(shape):
    # defining the input for the model
    #  this variable passes to the entire network
    """Input For the layers"""
    inputs = Input(shape)

    """Encoder"""
    #  importing the encoder
    #  include top basically means that if any classifier should be added at the end  of the network
    #  using the encoder for the segmentation task so there is not classification work is done i.e why the include top is set to False
    #  tensor input is the input definined above , i.e the input for the entire network
    #  the input passes to the resnet encoder
    #  first the input passes through the resnet50 encoder then we take the output i.e the image features
    encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    # More Info about conv4_block6_out: https://stackoverflow.com/questions/60234176/unpack-tf-keras-model-layer
    #  we are setting up the parameters of the image as conv4_block_out
    image_features = encoder.get_layer("conv4_block6_out").output
    #  now this image_features act as the output of the entire resnet50 architecture
    ASPP(image_features)


if __name__ == "__main__":
    deeplabv3_plus((512, 512, 3))
