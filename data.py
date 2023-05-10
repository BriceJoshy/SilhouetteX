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

