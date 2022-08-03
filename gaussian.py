import cv2 as cv
import numpy as np
import filtering

# Creating Gaussian blurring filter
def blurring(image, gaussian_kernel, stride=1):
    if gaussian_kernel is None:
        gaussian_kernel = 1 / 16 * np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])

    return filtering.convolution(image, gaussian_kernel, stride)
