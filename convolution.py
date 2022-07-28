
import numpy as np

# Convulating horizontally
def horizontal_convulation(A, G_x):
    m = A.shape[0]
    n = A.shape[1]
    
    anchor1 = np.empty((m-2, n-2))
    
    for i in range(m-2):
        for j in range(n-2):
            matrix_section = A[i: i+3, j: j+3]
            multiplied = matrix_section *  G_x
            total_sum = np.sum(multiplied)
            anchor1[i, j] = total_sum 
    return anchor1

# Convulating vertically            
def vertical_convulation(A, G_y):
    m = A.shape[0]
    n = A.shape[1]
    
    anchor2 = np.empty((m-2, n-2))
    for j in range(n-2):
        for i in range(m-2):
            matrix_section = A[i: i+3, j: j+3]
            multiplied = matrix_section *  G_y
            total_sum = np.sum(multiplied)
            anchor2[i, j] = total_sum 
    return anchor2

# Combining vertical and horizontal anchors
def combined(kernel1, kernel2):
    #G = np.sqrt(np.power(kernel1, 2) + np.power(kernel2, 2))
    G = np.absolute(kernel1) + np.absolute(kernel2)
    return G