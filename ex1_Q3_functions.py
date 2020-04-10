import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import unravel_index
from scipy import ndimage, signal
from numpy import pi, exp, sqrt


def myHarrisCornerDetector(IN, K, Threshold, use_grid):
    if len(IN.shape) >= 3:
        IN = cv2.cvtColor(IN, cv2.COLOR_BGR2GRAY)  # change to grayscale

    num_rows, num_cols = IN.shape[:2]

    # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
    s, k = 1, 2
    probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)]
    g = np.outer(probs, probs)

    # Shifting x by 1
    translation_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
    x_shifted = cv2.warpAffine(IN, translation_matrix, (num_cols, num_rows))

    # Shifting y by 1
    translation_matrix = np.float32([[1, 0, 0], [0, 1, 1]])
    y_shifted = cv2.warpAffine(IN, translation_matrix, (num_cols, num_rows))

    # Derivatives
    Ix = IN - x_shifted
    Yx = IN - y_shifted

    # S Calculating
    Sxx = signal.convolve2d(np.multiply(Ix, Ix), g, mode='same')
    Syy = signal.convolve2d(np.multiply(Yx, Yx), g, mode='same')
    Sxy = signal.convolve2d(np.multiply(Ix, Yx), g, mode='same')

    # Response image - R
    R = np.multiply(Sxx, Syy) - np.multiply(Sxy, Sxy) - K * np.multiply(Sxx + Syy, Sxx + Syy)
    R[R < Threshold] = 0

    # Slicing to 25X25 and taking highest corner only
    if use_grid:
        slice_size = 25
        for slicer in range(0, num_rows, slice_size):
            for slicec in range(0, num_cols, slice_size):
                Section = R[slicer:slicer + slice_size, slicec:slicec + slice_size]

                maxValue = np.amax(Section)
                maxIndex = unravel_index(Section.argmax(), Section.shape)
                R[slicer:slicer + slice_size, slicec:slicec + slice_size] = 0
                if maxValue > 0:
                    R[slicer + maxIndex[0], slicec + maxIndex[1]] = maxValue
    return R


def createCornerPlots(I1, I1_CORNERS, I2, I2_CORNERS):
    # general parameters
    radius = 2
    thickness = 2
    color = (0, 0, 255)
    I1c = cv2.transpose(I1_CORNERS)
    I2c = cv2.transpose(I2_CORNERS)
    # Converting back grayscale to color
    if len(I1.shape) < 3:
        I1 = cv2.cvtColor(I1, cv2.COLOR_GRAY2RGB)
    if len(I2.shape) < 3:
        I1 = cv2.cvtColor(I2, cv2.COLOR_GRAY2RGB)

    # Adding red circles around each corner detected
    for i in range(I1c.shape[0]):
        for j in range(I1c.shape[1]):
            pixel = I1c[i, j]
            if pixel.all() > 0:
                cv2.circle(I1, (i, j), radius + 1, color, thickness + 1, lineType=8, shift=0)

    for i in range(I2c.shape[0]):
        for j in range(I2c.shape[1]):
            pixel = I2c[i, j]
            if pixel.all() > 0:
                cv2.circle(I2, (i, j), radius, color, thickness, lineType=8, shift=0)

    # Displaying both images
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(cv2.cvtColor(I1, cv2.COLOR_BGR2RGB))
    axs[1].imshow(cv2.cvtColor(I2, cv2.COLOR_BGR2RGB))
    plt.show()

    """"
    imstack = cv2.resize(I1, (1000, 800))
    im2 = cv2.resize(I2, (1000, 800))
    imstack = np.hstack((imstack, im2))
    #cv2.imshow('Output Images', imstack)       # NOT SURE IF THIS IS CREATING A FIGURE OR NOT
    #cv2.waitKey(0)                             # SO WE USED SUBPLOT AND NOT CV2 IMSHOW
    # cv2.imwrite("ex1_ID1_ID2.jpg", imstack)
                                                    """


"""
##################################################################
# FILL IMAGE PATH
IN1 = cv2.imread('I1.jpg', 0)  # Image 1
IN2 = cv2.imread('I2.jpg')  # Image 2

# SET K AND THRESHOLD
K = 0.005  # K value should be 0.004-0.006
Threshold1 = 1  # Threshold to drop non-corner detections - image1
Threshold2 = 150  # Threshold to drop non-corner detections - image2
use_grid = True

I1_CORNERS = myHarrisCornerDetector(IN1, K, Threshold1, use_grid)
I2_CORNERS = myHarrisCornerDetector(IN2, K, Threshold2, use_grid)
createCornerPlots(IN1, I1_CORNERS, IN2, I2_CORNERS)
"""