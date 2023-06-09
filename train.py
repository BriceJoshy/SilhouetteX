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
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
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
        return x_image, y_mask

    # to include the numpy function in the tensor flow as these functions are outside ofthe tensorflow environment we use the tf.nump
    # to include the functions whuch are not included in the tensrflow we use the tf.numpy
    # First we mention the function to be used in tensorflow nd then the input then the specific datatypes
    #  and this tf.numpy_function return a variable/variables here x_image and y_mask
    x_image, y_mask = tf.numpy_function(
        _parse, [x_image, y_mask], [tf.float32, tf.float32]
    )

    # then we can set the shape of the x_image and y_mask
    #  we read the image as RGB image i.e the no: of channels is 3
    x_image.set_shape([H, W, 3])
    #  the number of channels is 1 as the mask is read as greyscale
    y_mask.set_shape([H, W, 1])

    return x_image, y_mask


# this will take an image and a mask and an optional batch
def tf_dataset(X_image, Y_mask, batch=2):
    #  the dataset will be a list
    dataset = tf.data.Dataset.from_tensor_slices((X_image, Y_mask))
    dataset = dataset.map(tf_parse)
    # it takes the individual path og the image and mask and give it to function x and why

    # after reading the individial files we'll create a batch of it
    dataset = dataset.batch(batch)
    # prefex some of them in the dataset
    # this is done to prevent the fetching of it after first batch has entered and i don'nt want the model to wait for the fetching and waste time
    # 10 batches  and each batches contain 2 images
    dataset = dataset.prefetch(10)
    return dataset


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

    training_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    validation_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    ## for checking the shape
    # for x_image, y_mask in training_dataset:
    #     print(x_image.shape, y_mask.shape)
    #     break

    """Working on the Model"""
    # Python compile() function takes source code as input and
    # returns a code object which is ready to be executed and which can later be executed by the exec() function
    model = deeplabv3_plus((H, W, 3))
    model.compile(
        loss=dice_loss,
        optimizer=Adam(learning_rate),
        metrics=[dice_coeff, iou, Recall(), Precision()],
    )
    # #  to get the summary of the model
    # model.summary()

    #  we need some callback to save the data
    callbacks = [
        # this is the place where the model is saved
        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
        ModelCheckpoint(model_save_path, verbose=0, save_best_only=True),
       
        # this is used to reduce the learning rate of the model
        # if the validation loss is not increased it decreases the learning rate with a factor
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        #  it is used to store all the matrices and the loss function for the training and vaidation process
        CSVLogger(csv__save_path),
        TensorBoard(),
        # if the model shows decreasing val_loss for 20 continues epoch then stop the model as it is not good to train the model then 
        # wastage of resourse is not good
        EarlyStopping(monitor = "val_loss",patience=20,restore_best_weights=False)
    ]

    # fits the training data into the model
    model.fit(
        training_dataset,
        epochs = num_epoch,
        validation_data= validation_dataset,
        callbacks=callbacks   
    )
