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
    # loading the Images by refering them with the variable X
    # loading the Masks by refering them with the variable Y
    # using the glob variable 
    # repective images and masks while zip() are not their respective ones so we need to sequence them
    # load all the images ending with the jpg extention For images
    """ For images """
    X =  glob(os.path.join(path,"images","*.jpg"))
    #  load all the images ending with the jpg extention For masks
    """ For maks """
    Y =  glob(os.path.join(path,"masks","*.png"))

    #  basically X and Y are lists
    #  the length of X and Y should be the same
    #  basically mapping the images to their corresponding masks
    #  More Imformations about zip() : https://realpython.com/python-zip-function/
    for x,y in zip(X,Y):
        print(x,y)

        #  read and save them here
        x = cv.imread(x)
        cv.im



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
    load_data(data_path)