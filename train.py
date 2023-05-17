#  training the deeplab_v3+ architecture
#  we will train it using the people segmentation dataset

import os
#  to remove the unecessary errors from the tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#  importing the libraries needed
import cv2 as cv
import numpy as np
# glob is used to extract the path (here images)
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint,CSVLogger,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import Adam
from keras.metrics import Recall,Precision
from model import deeplabv3_plus
from metrics import dice_coeff,dice_loss,iou

#  defining the global hyper parameters
""" Global parameters """
H = 512
W = 512

""" Creating a directory """
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffling """
#  used to shuffle the data given to the model
#  the random state should be the same as 42 as used before 
def shuffling(x,y):
    x,y = shuffle(x,y,random_state=42)
    return x,y

#  now the main function

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Creating a directory to store the files """
    create_directory("storage_files")

    """ Hyper parameters """
    # The batch size is a number of samples processed before the model is updated. 
    # The number of epochs is the number of complete passes through the training dataset.
    batch_size = 2
    # is a hyper-parameter used to govern the pace at which an algorithm updates or learns the values of a parameter estimate
    learning_rate = 1e-4
    # num_epochs indicates how many times will the input_fn return the whole batch
    # and steps indicates how many times the function should run.
    num_epoch = 20
    #  it is the path where you're saving you model
    model_save_path = os.path.join("storage_files","silhouetteX.h5")
    # A CSV file (Comma Separated Values file) is a type of plain text file that uses specific structuring to arrange tabular data
    #  More Info: https://realpython.com/python-csv/#:~:text=A%20CSV%20file%20(Comma%20Separated,given%20away%20by%20its%20name.
    csv__save_path = os.path.join("storage_files","data.csv")