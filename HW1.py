import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import cv2
from ex1_Q3_functions import *

# FILL IN YOUR ID
# ID1 = 200940500
# ID2 = 123123123

###############################################Part_3#######################################################

# YOU ARE REQUIRED TO WRITE THE FUNCTIONS:
#
# myHarrisCornerDetector
# createCornerPlots
#
# myHarrisCornerDetector accepts an image, and returns a binary matrix (same size as
# the original image dimensions) with '1' denoting corner pixels and '0'
# denoting all other pixels.
#
# createCornerPlots creates a plot with 2 subplots (on the left show I_1
#with the corner points and on the right show I2 with the corner points).
#
# IMPORTANT - DO NOT USE ANY FIGURE COMMANDS, ONLY SUBPLOT.
#             SETTING ANOTHER figure COMMAND WILL MESS UP THE SAVING
#             PROCESS AND YOU WILL LOSE POINTS IF THAT HAPPENS!


# LOAD CHECKERBOARD IMAGE
I1 = cv2.imread('I1.jpg', 0)  # Read image as grayscale
# LOAD ATTACHED IMAGE
I2 = cv2.imread('I2.jpg')

# Harris Corner Detector Parameters, you may change them
K = 0.005
FIRST_THRESHOLD = 1
SECOND_THRESHOLD = 150
use_grid = True

# CALL YOUR FUNCTION TO FIND THE CORNER PIXELS
I1_CORNERS = myHarrisCornerDetector(I1, K, FIRST_THRESHOLD, use_grid)
I2_CORNERS = myHarrisCornerDetector(I2, K, SECOND_THRESHOLD, use_grid)

# CALL YOUR FUNCTION TO CREATE THE PLOT
createCornerPlots(I1, I1_CORNERS, I2, I2_CORNERS)
