
import cv2 as cv
import numpy as np
import os


path = r'/Users/arina/Desktop/LabProject/images/lena.png'
directory = r'/Users/arina/Desktop/LabProject/images'


# Тhis method loads an image from the specified file
img = cv.imread(path)

# Converts an image from one color space to another
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
filename = 'gray.jpeg'
# cv.imshow(filename, gray)

# Gaussian bluring
Gaussian = cv.GaussianBlur(gray, (3, 3), 0)
print("Gaussian shape: ", Gaussian.shape)
print(Gaussian)
# filename = 'Gaussian.jpeg'
# cv.imshow(filename, Gaussian)


scale = 1
delta = 0
ddepth = cv.CV_16S

grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale,
                  delta=delta, borderType=cv.BORDER_DEFAULT)

abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)

filename = 'abs_grad_x.jpeg'
cv.imshow(filename, abs_grad_x)
cv.imwrite(filename, abs_grad_x)

filename = 'abs_grad_y.jpeg'
cv.imshow(filename, abs_grad_y)
cv.imwrite(filename, abs_grad_y)

filename = 'edgeweighted.jpeg'
cv.imshow(filename, cv.addWeighted(abs_grad_x, 0.1, abs_grad_y, 0.9, 0))
cv.imwrite(filename, cv.addWeighted(abs_grad_x, 0.1, abs_grad_y, 0.9, 0))

cv.waitKey(0)
