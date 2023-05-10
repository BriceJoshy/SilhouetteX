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
def load_data(path,split=0.1):
    """ Loading images and masks """
    # loading the Images by refering them with the variable X
    # loading the Masks by refering them with the variable Y
    # using the glob variable 
    # repective images and masks while zip() are not their respective ones so we need to sequence them
    """ Sorting """
    #  i.e why the "sorted" function is used
    # load all the images ending with the jpg extention For images
    """ For images """
    X =  sorted(glob(os.path.join(path,"images","*.jpg")))
    #  load all the images ending with the jpg extention For masks
    """ For maks """
    Y =  sorted(glob(os.path.join(path,"masks","*.png")))


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
    split_size = int(len(X) * split)

    # splitting the image and masks
    # providing X and Y ,basically the list of all images
    # make sure the random state is same or else it would not be split the same
    train_x, test_x = train_test_split(X, test_size=split_size,random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size,random_state=42)

    #  returning the training and testing set
    return (train_x,test_x),(train_y,test_y)

""" Main Function """
# The __name__ variable merely holds the name of the module or script unless the current module is executing,
# __main__ is the name of the environment where top-level code is run.
#  “Top-level code” is the first user-specified Python module that starts running.
if __name__=="__main__":
    """ Seeding the environment """
    #  the randomness present between the numpy , we will seed it so that i have the same randomness every time
    # More Info: https://medium.com/geekculture/the-story-behind-random-seed-42-in-machine-learning-b838c4ac290a#:~:text=seed()%20%3F-,random.,generator%20with%20the%20given%20value.
    np.random.seed(42)
    # Reproducibility is a very important concept that ensures that anyone who re-runs the code gets the exact same outputs.

    """ Loading the DataSet """
    #  path for the data set (below)
    data_path = "people_segmentation"
    # function load data, this function will take the data path and returns the training and testing the masks and images
    (train_x,test_x),(train_y,test_y) = load_data(data_path)

    """ Checking the length of the training and testing"""

    print(f"Train:\t{len(train_x)} - {len(train_y)}")
    print(f"Train:\t{len(test_x)} - {len(test_y)}")

    """ Applying the data augmentation for training"""
    # on testing we wont appy