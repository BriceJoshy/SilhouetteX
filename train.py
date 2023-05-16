#  training the deeplab_v3+ architecture
#  we will train it using the people segmentation dataset

import os
#  to remove the unecessary errors from the tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#  importing the libraries needed
import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow import keras
