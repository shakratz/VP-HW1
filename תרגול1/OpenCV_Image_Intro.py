import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

'''
DANIEL KIGLI - TAU - VIDEO PROCESSING 2020
'''

img_BGR = cv2.imread("image.jpg")
assert(img_BGR is not None)  # We want to check that we actually read the image

# OpenCV works in BGR color space, and matplotlib works in RGB
img_in_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

# Visualize image using OpenCV
cv2.imshow("Original image - OpenCV imshow", img_BGR)

# Visualize image using matplotlib
fig = plt.figure()
plt.imshow(img_BGR)
fig.suptitle("Original Image - BGR instead of RGB\nMatplotlib imshow")
# cv2.waitKey()  # cv2.waitKey(WAIT_KEY_T)

# Convert image to grayscale
img_in_grayscale = cv2.cvtColor(img_in_RGB, cv2.COLOR_RGB2GRAY)

# Show differences with color spaces
fig, axs = plt.subplots(1, 3)
axs[0].imshow(img_BGR)
axs[0].set_title('Orig Img - BGR')
axs[1].imshow(img_in_RGB)
axs[1].set_title('Orig Img - RGB')
axs[2].imshow(img_in_grayscale, cmap='gray')
axs[2].set_title('Orig Img - Grayscale')
plt.show(block=False)

# Apply filters on image
# I will show Soble filter, for more information: https://en.wikipedia.org/wiki/Sobel_operator
deriv_X_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
gray_deriv_X = signal.convolve2d(img_in_grayscale, deriv_X_filter, mode='same')  # Keep output image with same size.
deriv_Y_filter = deriv_X_filter.copy().transpose()
gray_deriv_Y = signal.convolve2d(img_in_grayscale, deriv_Y_filter, mode='same')  # Keep output image with same size.
soble_filter_edges = np.sqrt(np.square(gray_deriv_X) + np.square(gray_deriv_Y))

# Visualize
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(img_in_grayscale, cmap='gray')
axs[0, 0].set_title('Orig Img - Grayscale')
axs[0, 1].imshow(gray_deriv_X, cmap='gray')
axs[0, 1].set_title('X derivative')
axs[1, 0].imshow(gray_deriv_Y, cmap='gray')
axs[1, 0].set_title('Y derivative')
axs[1, 1].imshow(soble_filter_edges, cmap='gray')
axs[1, 1].set_title('Sobel Filter')
fig.suptitle("Applying filters on image example")
plt.show(block=True)  # Usually not a good idea!!!!

