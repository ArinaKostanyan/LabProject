
import cv2 as cv
import numpy as np
import math


#  Defaults 3x3 kernels for vertical and horizontal conc=volution
kernel1 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])
kernel2 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])


# Convulating horizontally
def horizontal_convulation(A, G_x = None, stride = 1):
    m = A.shape[0]
    n = A.shape[1]
    
    if G_x == None:
        G_x = kernel1
    
    kernel_row = G_x.shape[0]
    kernel_col = G_x.shape[1]
    
    anchor1 = np.zeros((m - kernel_row + 1, n - kernel_col + 1))
    
    for i in range(0, m - kernel_row + 1, stride):
        for j in range(0, n - kernel_col + 1, stride):
            matrix_section = A[i: i + kernel_row, j: j + kernel_col]
            
            multiplied_h = matrix_section *  G_x
            horizon_sum = np.sum(multiplied_h)
            
            anchor1[i][j] = horizon_sum
            
    return cv.convertScaleAbs(anchor1)

# Convulating vertically            
def vertical_convulation(A, G_y = None, stride = 1):
    m = A.shape[0]
    n = A.shape[1]
    
    if G_y == None:
        G_y = kernel2
    
    kernel_row = G_y.shape[0]
    kernel_col = G_y.shape[1]
    
    anchor2 = np.zeros((m-kernel_row+1, n-kernel_col+1))
    
    for j in range(0, n - kernel_col + 1, stride):
        for i in range(0, m - kernel_row + 1, stride):
            matrix_section = A[i: i+kernel_row, j: j+kernel_col]
            
            multiplied_v = matrix_section *  G_y
            vert_sum = np.sum(multiplied_v)
            
            anchor2[i, j] = vert_sum 
    
    return cv.convertScaleAbs(anchor2)


def combined(A, G_x = None, G_y = None, stride = 1):
    m = A.shape[0]
    n = A.shape[1]
    
    if G_x == None:
        G_x = kernel1
    if G_y == None:
        G_y = kernel2    
    
    kernel_row = G_x.shape[0]
    kernel_col = G_x.shape[1]
    
    anchor = np.zeros((m - kernel_row + 1, n - kernel_col + 1))
    
    for i in range(0, m - kernel_row + 1, stride):
        for j in range(0, n - kernel_col + 1, stride):
            matrix_section = A[i: i + kernel_row, j: j + kernel_col]
            
            multiplied_h = matrix_section *  G_x
            horizon_sum = np.sum(multiplied_h)
            
            multiplied_v = matrix_section *  G_y
            vert_sum = np.sum(multiplied_v)
            
            result = np.absolute(horizon_sum) + np.absolute(vert_sum)
            result = math.sqrt(horizon_sum**2 + vert_sum**2)
            
            anchor[i][j] = result
            
    return cv.convertScaleAbs(anchor)
