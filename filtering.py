import cv2 as cv
import numpy as np
import math

"""#  Defaults 3x3 kernels for vertical and horizontal convolution
kernel1 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
kernel2 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])"""


# Horizontal convolution
def convolution(image, kernel, stride=1):
    m = image.shape[0]
    n = image.shape[1]

    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]

    anchor = np.zeros((m - kernel_row + 1, n - kernel_col + 1))

    for i in range(0, m - kernel_row + 1, stride):
        for j in range(0, n - kernel_col + 1, stride):
            matrix_section = image[i: i + kernel_row, j: j + kernel_col]

            multiplied = matrix_section * kernel
            summa = np.sum(multiplied)

            anchor[i][j] = summa

    return cv.convertScaleAbs(anchor)




def combinacia(anchor1, anchor2):
    return cv.convertScaleAbs(np.absolute(anchor1) + np.absolute(anchor2))