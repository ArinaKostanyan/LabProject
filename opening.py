
import cv2 as cv
import numpy as np
import os
import convolution 


path = r'/Users/arina/Desktop/LabProject/images/images.jpeg'
directory = r'/Users/arina/Desktop/LabProject/images'


# Тhis method loads an image from the specified file
img = cv.imread(path)

# Gaussian bluring
Gaussian = cv.GaussianBlur(img, (3, 3), 0)

# Converts an image from one color space to another
gray = cv.cvtColor(Gaussian,cv.COLOR_BGR2GRAY)


# Change the current directory to specified directory 
os.chdir(directory)


anchor1 = convolution.horizontal_convulation(gray)
anchor2 = convolution.vertical_convulation(gray)
anchor = convolution.combined(gray)

anchor_weighted = cv.addWeighted(anchor1,0 ,  anchor2, 1, 0)
filename = 'anchorweighted_gaus.jpeg'
cv.imshow(filename, anchor_weighted)
cv.imwrite(filename, anchor_weighted)

scale = 1
delta = 0
ddepth = cv.CV_16S

grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)


# Vertical and horizontal convolution printing
"""filename = 'convertScaleAbs(anchor1).jpeg_gaus'
cv.imshow(filename, anchor1)
# Saving the image
cv.imwrite(filename, anchor1)

filename = 'convertScaleAbs(anchor2)_gaus.jpeg'
cv.imshow(filename, anchor2)
cv.imwrite(filename, anchor2)
"""


"""filename = 'edgeweighted_gaus.jpeg'
cv.imshow(filename, cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0))
cv.imwrite(filename, cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0))
"""

filename = 'Edge_detected.jpeg'
cv.imshow(filename, anchor)
cv.imwrite(filename, anchor)
cv.waitKey(0)