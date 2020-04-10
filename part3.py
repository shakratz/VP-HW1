import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy import unravel_index
from scipy import ndimage, signal


def myHarrisCornerDetector(IN, K, Threshold):
    imgcolor = cv2.imread(IN)
    img = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    num_rows, num_cols = img.shape[:2]
    g = np.ones((5, 5), dtype=int)
    slice_size = 25
    # Shifting x by 1
    translation_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
    x_shifted = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    # Shifting y by 1
    translation_matrix = np.float32([[1, 0, 0], [0, 1, 1]])
    y_shifted = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))

    # Derivatives
    Ixs = img - x_shifted
    translation_matrix = np.float32([[1, 0, -1], [0, 1, 0]])
    Ix = cv2.warpAffine(Ixs, translation_matrix, (num_cols, num_rows))

    Yxs = img - y_shifted
    translation_matrix = np.float32([[1, 0, 0], [0, 1, -1]])
    Yx = cv2.warpAffine(Yxs, translation_matrix, (num_cols, num_rows))


    #cv2.imshow('R', Yx)
    #cv2.waitKey(0)
    #cv2.imshow('R', Ix)
    #cv2.waitKey(0)
    #cv2.imshow('R', imgcolor)
    #cv2.waitKey(0)

    # S
    Sxx = signal.convolve2d(np.multiply(Ix, Ix), g, mode='same')
    Syy = signal.convolve2d(np.multiply(Yx, Yx), g, mode='same')
    Sxy = signal.convolve2d(np.multiply(Ix, Yx), g, mode='same')
    # Response image - R
    R = np.multiply(Sxx, Syy) - np.multiply(Sxy, Sxy) - K * np.multiply(Sxx + Syy, Sxx + Syy)
    R[R < Threshold] = 0

    cv2.imshow('R', R)
    cv2.waitKey(0)

    # Slicing to 25X25 and taking highest corner only
    slices_r = int(np.math.ceil(num_rows / slice_size))
    slices_c = int(np.math.ceil(num_cols / slice_size))
    for slicer in range(0, num_rows, slice_size):
        for slicec in range(0, num_cols, slice_size):
            Section = R[slicer:slicer + slice_size, slicec:slicec + slice_size]

            maxValue = np.amax(Section)
            maxIndex = unravel_index(Section.argmax(), Section.shape)
            R[slicer:slicer + slice_size, slicec:slicec + slice_size] = 0
            if maxValue > 0:
                R[slicer + maxIndex[0], slicec + maxIndex[1]] = maxValue
    cv2.imshow('R', R)
    cv2.waitKey(0)
    return R


def createCornerPlots(I1, I1_CORNERS, I2, I2_CORNERS):
    # general parameters
    radius = 2
    thickness = 2
    color = (0, 0, 255)
    I1c = cv2.transpose(I1_CORNERS)
    I2c = cv2.transpose(I2_CORNERS)
    I1im = cv2.imread(I1)
    I2im = cv2.imread(I2)

    # Adding red circles around each corner detected
    for i in range(I1c.shape[0]):
        for j in range(I1c.shape[1]):
            pixel = I1c[i, j]
            if pixel.all() > 0:
                cv2.circle(I1im, (i, j), radius, color, thickness, lineType=8, shift=0)

    for i in range(I2c.shape[0]):
        for j in range(I2c.shape[1]):
            pixel = I2c[i, j]
            if pixel.all() > 0:
                cv2.circle(I2im, (i, j), radius, color, thickness, lineType=8, shift=0)

    # Displaying both images
    imstack = cv2.resize(I1im, (1000, 800))
    im2 = cv2.resize(I2im, (1000, 800))
    imstack = np.hstack((imstack, im2))
    cv2.imshow('stack', imstack)
    cv2.waitKey(0)


#Calling the function
IN1 = "I1.jpg" # Image 1
IN2 = "I2.jpg" # Image 2
K = 0.05 # K value should be 0.04-0.06
Threshold1 = 50     # Threshold to drop non-corner detections - image1
Threshold2 = 50     # Threshold to drop non-corner detections - image2
I1_CORNERS = myHarrisCornerDetector(IN1, K, Threshold1)
I2_CORNERS = myHarrisCornerDetector(IN2, K, Threshold2)
createCornerPlots(IN1, I1_CORNERS, IN2, I2_CORNERS)
