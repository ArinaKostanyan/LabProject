import cv2 as cv
import numpy as np


kernel_g = 1/16 * np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]])

def filter(image, kernel, stride ):

    m = image.shape[0]
    n = image.shape[1]
    
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
    
    anchor0 = np.zeros((m - kernel_row + 1, n - kernel_col + 1))
    
    for i in range(0, m - kernel_row + 1, stride):
        for j in range(0, n - kernel_col + 1, stride):
            matrix_section = image[i: i + kernel_row, j: j + kernel_col]
            
            multiplied = matrix_section *  kernel 
            sum = np.clip(np.sum(multiplied), 0, 255)
            
            anchor0[i][j] = sum 
    
    return cv.convertScaleAbs(anchor0)
    

# Creating Gaussian blurring
def blurring(image, kernel = None, stride = 1):
    if kernel == None:
        kernel = kernel_g
        
    blurred = filter(image, kernel, stride)
    
    return (blurred)