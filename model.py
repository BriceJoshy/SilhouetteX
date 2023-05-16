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
    Concatenate,
)
from keras.layers import (
    AveragePooling2D,
    GlobalAveragePooling2D,
    UpSampling2D,
    Reshape,
    Input,
    Dense,
)
from keras.applications import ResNet50
from keras.models import Model
import tensorflow as tf

"""Squeeze and Excitation Network (SENet)"""


#  we take the inputs and the ratio
#  ratio = 8 means that to compress the features as 8
def SqueezeAndExcite(inputs, ratio=8):
    #  making a copy of the inputs
    init = inputs
    #  taking the filters from the shape as the filters are the last axis element eg:(none,32,32,1024)
    # ⬆️
    filters = init.shape[-1]
    # setting the shape using the filters taken above
    squeeze_excite_shape = (1, 1, filters)

    # To get the global features of all the channels
    # so that we get an global embedding of each one respectively
    sqeeze_excite = GlobalAveragePooling2D()(init)

    #  reshaping it
    sqeeze_excite = Reshape(squeeze_excite_shape)(sqeeze_excite)

    # Followed by a dense layer
    # then Dense connections are done
    # //filters mean sqeeze or compress the features
    # using the 'relu activation and kernal_initializer'
    sqeeze_excite = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(sqeeze_excite)

    #  again a dense layer but change in the number of filters
    sqeeze_excite = Dense(filters // ratio,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,)(sqeeze_excite)


"""Function for Deeplabv3+"""


#  Atrous Spacial Pyramid Pooling for deeplabv3+
#  More Info: https://developers.arcgis.com/python/guide/how-deeplabv3-works/
def ASPP(inputs):
    """Image Pooling"""
    # Average pooling is  downsizing the image wrt the dimensions of it
    # Avreage pooling is done for the input
    shape = inputs.shape
    #  a pool size of about 32 by 32 as we print the shape we got (none,32,32,1024)
    # checking the shape
    # print(inputs.shape)
    #  now the feature map is converted to 1:1 by 1024
    # setting the input as 'inputs'
    y1_input = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    #  now the featuer map is (none,1,1,1024)
    # checking the shape
    # print(y1.shape)

    #  now we will upsample and convolution layer
    #  256 as features and size as 1 padding is same as we dont want to change the shape
    # More Info about convo2d: https://keras.io/api/layers/convolution_layers/convolution2d/
    # use_bais is a boolean used to see if the layer used a bias vector
    # More Info about: https://deepai.org/machine-learning-glossary-and-terms/bias-vector
    #  and y1 as input
    y1_input = Conv2D(filters=256, kernel_size=1, padding="same", use_bias=False)(
        y1_input
    )

    """we do the batch_normalization where y1 as input:"""
    # Layer that normalizes its inputs.
    #  Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
    # More info: https://keras.io/api/layers/normalization_layers/batch_normalization/
    y1_input = BatchNormalization()(y1_input)

    # then followed by activation: Applies an activation function to an output
    # relu function: Applies the rectified linear unit activation function.
    # More info about relu: https://keras.io/api/layers/activations/
    # More info about relu function : https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning
    # and y1 as input for activation
    y1_input = Activation("relu")(y1_input)

    # still the shape is 1x1 so we are going to upsample the shape
    # How much we want to upsample i.e same as pooling
    # More Info about bilinear interpolation: https://web.pdx.edu/~jduh/courses/geog493f09/Students/W6_Bilinear%20Interpolation.pdf
    # More info about upsampling: https://keras.io/api/layers/reshaping_layers/up_sampling2d/
    # Input as y1
    y1_input = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1_input)

    # printing the shape of y1
    # the image channels changed to 256 we get (none 32,32,256)
    # checking the shape
    # print(y1.shape)

    # After image Pooling
    """ 1X1 Convolutions """
    #  the same thing with different name 'y2'
    #  the image feature as the input for the first one
    y2_input = Conv2D(filters=256, kernel_size=1, padding="same", use_bias=False)(
        inputs
    )
    y2_input = BatchNormalization()(y2_input)
    y2_input = Activation("relu")(y2_input)

    # after the 1x1 convolution is done it is followed by the three
    # Where the dialation rate would be applied
    #  dialation bassically increases the recepty field of the convolution
    # and the dialtion rate is 6,12,18 sequentially
    #  here the color size is changed to 3
    """ 3x3 Convolutions rate = 6 """
    y3_input = Conv2D(
        filters=256, kernel_size=3, padding="same", use_bias=False, dilation_rate=6
    )(inputs)
    y3_input = BatchNormalization()(y3_input)
    y3_input = Activation("relu")(y3_input)

    """ 3x3 Convolutions rate = 12 """
    y4_input = Conv2D(
        filters=256, kernel_size=3, padding="same", use_bias=False, dilation_rate=12
    )(inputs)
    y4_input = BatchNormalization()(y4_input)
    y4_input = Activation("relu")(y4_input)

    """ 3x3 Convolutions rate = 18 """
    y5_input = Conv2D(
        filters=256, kernel_size=3, padding="same", use_bias=False, dilation_rate=18
    )(inputs)
    y5_input = BatchNormalization()(y5_input)
    y5_input = Activation("relu")(y5_input)

    #  now we are going to concatinate all these features
    """ Concatination Of features"""
    y_input = Concatenate()([y1_input, y2_input, y3_input, y4_input, y5_input])

    #  this is agoin followed by 1x1 convolution
    """ 1x1 Convolution"""
    y_input = Conv2D(filters=256, kernel_size=1, padding="same", use_bias=False)(
        y_input
    )
    y_input = BatchNormalization()(y_input)
    y_input = Activation("relu")(y_input)

    return y_input


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
    aspp_out_a = ASPP(image_features)
    aspp_out_a = UpSampling2D((4, 4), interpolation="bilinear")(aspp_out_a)
    # we need to upsample the aspp_put four times "(None, 32, 32, 32)"
    # we can see that the shape is changed to (None, 128, 128, 256)
    # checking the shape
    # print(aspp_out.shape)

    #  now we are going to extract the low level features using this upsampled image
    aspp_out_b = encoder.get_layer("conv2_block2_out").output
    """1x1 Convolutions"""
    # Followed by convolutions
    aspp_out_b = Conv2D(filters=48, kernel_size=1, padding="same", use_bias=False)(
        aspp_out_b
    )
    aspp_out_b = BatchNormalization()(aspp_out_b)
    aspp_out_b = Activation("relu")(aspp_out_b)

    """concatination"""
    # Then by the concatination
    aspp_out = Concatenate()([aspp_out_a, aspp_out_b])
    # checking the shape
    # it changed to (None, 128, 128, 304)
    # print(aspp_out.shape)

    """3x3 convolutions"""
    #  followed by some more convolutions
    aspp_out = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(
        aspp_out
    )
    aspp_out = BatchNormalization()(aspp_out)
    aspp_out = Activation("relu")(aspp_out)

    #  checking the shape
    # print(aspp_out.shape)

    """3x3 convolutions"""
    # Followed by 3x3 convolutions
    aspp_out = Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(
        aspp_out
    )
    aspp_out = BatchNormalization()(aspp_out)
    aspp_out = Activation("relu")(aspp_out)

    """Upsampling"""
    #  Followed by upsampling
    aspp_out = UpSampling2D((4, 4), interpolation="bilinear")(aspp_out)

    # followed by 1x1 convolution layer
    """1x1 convolutions"""
    aspp_out = Conv2D(filters=1, kernel_size=1)(aspp_out)
    #  here we are doing binary segmentation
    aspp_out = Activation("sigmoid")(aspp_out)
    # the shape changed to (None, 512, 512, 1)
    # this is the output shape of our predicted mask
    # print(aspp_out.shape)

    """building the model"""
    model = Model(inputs, aspp_out)
    return model


if __name__ == "__main__":
    model = deeplabv3_plus((512, 512, 3))
    model.summary()
    # the "None" representing the shape is "batch size"
    # in the terminal it shows the correct architecture in all the stages like he concatinated features
    # and all the entire summary is shown the terminal
    # and this entire architecture is the "Deep Lab" with "resnet50" as the pre-trained architecture
    # now we need channel-wise attension mechanism
    # so we use Squeeze and Excitation Network (SENet)
    # it is a mechanism to imporove the existing CNN networks in creating nore channel interdependencies
    # i.e improves the efficinecy of the neural network
    # this method is maily used to decrease the computational usage
    # More info: https://idiotdeveloper.com/squeeze-and-excitation-networks/
