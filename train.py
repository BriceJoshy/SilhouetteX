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
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from model import deeplabv3_plus
from metrics import dice_coeff, dice_loss, iou
import data

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
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


""" Reading the Image"""


def reading_image(path):
    #  decode the path of the image as string
    path = path.decode()
    #  now reading the image as rgb
    x_image = cv.imread(path, cv.IMREAD_COLOR)
    #  normalizing the image by dividing the image by 255
    x_image = x_image / 255.0
    # converting the image to float32
    x_image = x_image.astype(np.float32)
    return x_image


""" Reading the Masks"""


def reading_mask(path):
    #  decode the path of the image as string
    path = path.decode()
    #  now reading the image as rgb
    y_mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
    # no need to normalize the image as the image is already in the range f 0  and 1 cuz of the greyscale
    # converting the image to float32
    y_mask = y_mask.astype(np.float32)
    # More info: https://www.educative.io/answers/what-is-the-numpyexpanddims-function-in-numpy
    # Expanding the last axis
    y_mask = np.expand_dims(y_mask, axis=-1)
    return y_mask


def tf_parse(x_image, y_mask):
    def _parse(x_image, y_mask):
        x_image = reading_image(x_image)
        y_mask = reading_mask(y_mask)
        return x_image,y_mask
    # to include the numpy function in the tensor flow as these functions are outside of the tensorflow environment we use the tf.nump
    # to include the functions whuch are not included in the tensrflow we use the tf.numpy
    

""" Load Data """


def load_data(path):
    #  x refers to the image and y refers to the masks
    #  x refers to the features and the y refers to the labels\
    X_image = sorted(glob(os.path.join(path, "image", "*.png")))
    Y_mask = sorted(glob(os.path.join(path, "mask", "*.png")))
    return X_image, Y_mask


#  now the main function
""" Main Function """
if __name__ == "__main__":
    """Seeding"""
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
    model_save_path = os.path.join("storage_files", "silhouetteX.h5")
    # A CSV file (Comma Separated Values file) is a type of plain text file that uses specific structuring to arrange tabular data
    #  More Info: https://realpython.com/python-csv/#:~:text=A%20CSV%20file%20(Comma%20Separated,given%20away%20by%20its%20name.
    csv__save_path = os.path.join("storage_files", "data.csv")

    """ DataSet """
    # we will be using the aumented data we processed
    # specifying the dataset path first
    dataset_path = "new_data_generated"
    training_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "test")

    #  we are going to load the data using the load dataset
    train_x, train_y = load_data(training_path)
    # we are goin to shuffle the different data in between
    train_x, valid_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(validation_path)

    print(f"Training:\t{len(train_x)} - {len(train_y)}")
    print(f"validation:\t{len(valid_x)} - {len(valid_y)}")
