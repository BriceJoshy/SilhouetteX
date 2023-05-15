#  This is the data processing part
#  we are going to load the data and split it and data augmentation on the training part
#  and leave the validation as it is

#  importing the libraries
# importing the os
import os
import numpy as np
import cv2 as cv

# glob is used to extract the path (here images)
from glob import glob

# for progress bar as the progress is not visible for us(how much are done and how much are left)
from tqdm import tqdm

# Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python
#  More Info : https://www.tutorialspoint.com/scikit_learn/scikit_learn_introduction.htm#:~:text=Scikit%2Dlearn%20(Sklearn)%20is,a%20consistence%20interface%20in%20Python.
from sklearn.model_selection import train_test_split

# for augmentation using the augmentation library below
# Data augmentation is a technique of artificially increasing the training set by creating modified copies of a dataset using existing data
#  functions stated after the import keyword
# More Info: https://www.datacamp.com/tutorial/complete-guide-data-augmentation
from albumentations import (
    HorizontalFlip,
    GridDistortion,
    OpticalDistortion,
    ChannelShuffle,
    CoarseDropout,
    CenterCrop,
    Crop,
    Rotate,
)

""" Creating a directory """


#  creating a function for creating directory
# for creating a directory by itself
# using a os module
def create_directory(path):
    #  checking if the path exist if not create the path
    if not os.path.exists(path):
        os.makedirs(path)


"""Load Data function"""


# passing the path and argument split as 0.1
# 0.1 means out of the complete data we will be using the 1% of the data for the testing and validation purpose
# here the testing and the validation split is the same
def loading_data(path, split=0.1):
    """Loading images and masks"""
    # loading the Images by refering them with the variable X
    # loading the Masks by refering them with the variable Y
    # using the glob variable
    # repective images and masks while zip() are not their respective ones so we need to sequence them
    """ Sorting """
    #  i.e why the "sorted" function is used
    # load all the images ending with the jpg extention For images
    """ For images """
    X_image = sorted(glob(os.path.join(path, "images", "*.jpg")))
    #  load all the images ending with the jpg extention For masks
    """ For maks """
    Y_mask = sorted(glob(os.path.join(path, "masks", "*.png")))

    """Only for Chacking the image and masks are their corresponding pairs"""
    # #  basically X and Y are lists
    # #  the length of X and Y should be the same
    # #  basically mapping the images to their corresponding masks nut its wrong if sorted function is not used
    # #  More Imformations about zip() : https://realpython.com/python-zip-function/
    # for x,y in zip(X,Y):
    #     print(x,y)

    #     #  read and save them here
    #     x = cv.imread(x)
    #     cv.imwrite("x.png",x)

    #     #  read and save masks here
    #     y = cv.imread(y)
    #     cv.imwrite("y.png",y*255) # making the mask white for checking

    #     break

    """splitting the data into testing and training"""
    #  split size will contain the 1% of x
    spliting_size = int(len(X_image) * split)

    # splitting the image and masks
    # providing X and Y ,basically the list of all images
    # make sure the random state is same or else it would not be split the same
    train_x, test_x = train_test_split(
        X_image, test_size=spliting_size, random_state=42
    )
    train_y, test_y = train_test_split(Y_mask, test_size=spliting_size, random_state=42)

    #  returning the training and testing set
    return (train_x, test_x), (train_y, test_y)


def augment_data(images, masks, save_path, augment=True):
    # setting the item height and wifth , we'll set it because fixed data needed (size)
    H = 512
    W = 512

    #  loop over the images
    # tqdm used for the progress bar, one variable "total"how much the loop would run
    #  for saving the images we need a name
    #  splitting it using the slash
    for x, y in tqdm(zip(images, masks), total=len(images)):
        # print(x,y)
        # break
        """Extract the name"""
        name = x.split("\\")[-1].split(".")[0]
        # print(name)

        """Read the image and mask"""
        #  reading the image as rgb
        x = cv.imread(x, cv.IMREAD_COLOR)
        #  reading the mask as rgb
        y = cv.imread(y, cv.IMREAD_COLOR)

        #  after loading the library
        if augment == True:
            #  if True something will definitely happn and incrase the list with more array of images
            # Using the horizondal flip for data augmentation
            # More Info : https://hasty.ai/docs/mp-wiki/augmentations/horizontal-flip#:~:text=Horizontal%20Flip%20explained,-As%20you%20might&text=To%20define%20the%20term%2C%20Horizontal,horizontally%20along%20the%20y%2Daxis.
            # p is the probability as it is set to 1.0 i surely want to apply horizondal flip
            aug = HorizontalFlip(p=1.0)
        else:
            # If False , then we resize the original image and save them
            X = [x]
            Y = [y]

        #  as the images are not of equal size (portrait & landscape)
        #  there is need of resizing these images
        # the images is compressed unevenly if resized
        #  looping them
        for i_image, m_mask in zip(X, Y):
            # Center crop is cropping an image from center which gives an equal padding on both sides vertically and horizontally
            # More Info: https://hasty.ai/docs/mp-wiki/augmentations/center-crop
            #  resizing the images and mask causes the uneven compressing

            try:
                """Centre Cropping"""
                # Height - defines the height of the newly cropped image in pixels;
                # Width - defines the width of the newly cropped image in pixels;
                # Probability - the probability that the transformation will be applied to an image.
                aug = CenterCrop(H, W, p=1.0)
                augmented = aug(image=i_image, mask=m_mask)
                i_image = augmented["image"]
                m_mask = augmented["mask"]

            except Exception as e:
                """Incase the the Centre Cropping doesn't work do the resize operation"""
                i_image = cv.resize(i_image, (W, H))
                m_mask = cv.resize(m_mask, (W, H))

            # saving the images names as temporary image and mask name
            temp_image_name = f"{name}.png"
            temp_mask_name = f"{name}.png"

            #  setting the path of the images and masks for saving
            image_path = os.path.join(save_path, "image", temp_image_name)
            mask_path = os.path.join(save_path, "mask", temp_mask_name)

            #  saving the images and masks generated
            cv.imwrite(image_path, i_image)
            cv.imwrite(mask_path, m_mask)

        break


""" Main Function """
# The __name__ variable merely holds the name of the module or script unless the current module is executing,
# __main__ is the name of the environment where top-level code is run.
#  “Top-level code” is the first user-specified Python module that starts running.
if __name__ == "__main__":
    """Seeding the environment"""
    # the randomness present between the numpy , we will seed it so that i have the same randomness every time
    # More Info: https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a#:~:text=seed()%20%3F-,random.,generator%20with%20the%20given%20value.
    np.random.seed(42)
    # Reproducibility is a very important concept that ensures that anyone who re-runs the code gets the exact same outputs.

    """ Loading the DataSet """
    #  path for the data set (below)
    data_path = "people_segmentation"
    # function load data, this function will take the data path and returns the training and testing the masks and images
    (train_x, test_x), (train_y, test_y) = loading_data(data_path)

    """ Checking the length of the training and testing"""

    print(f"Train:\t{len(train_x)} - {len(train_y)}")
    print(f"Train:\t{len(test_x)} - {len(test_y)}")

    """ Creating directories to save the augmeneted data for images and masks seperatly"""
    create_directory("new_data_generated/train/image/")
    create_directory("new_data_generated/train/mask/")
    create_directory("new_data_generated/test/image/")
    create_directory("new_data_generated/test/mask/")

    """ Applying the data augmentation for training"""
    # on testing we wont appy
    #  creating a function for augmenting
    #  first it would take the images either for the training or for the mask
    # then it would take the path where thr images is needed to be saved
    # last a booblean variable cld augment,
    # if it is true then appy augmnetation if false dont apply

    augment_data(train_x, train_y, "new_data/train/", augment=True)
