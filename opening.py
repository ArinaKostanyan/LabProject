import cv2 as cv
import numpy as np
import convolution 
from PIL import Image as im
# Тhis method loads an image from the specified file
img = cv.imread('/Users/arina/python_p/LabProject/images/images.jpeg')

# Converts an image from one color space to another
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

G_x = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
G_y = np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]])

anchor1 = convolution.horizontal_convulation(gray, G_x)
anchor2 = convolution.vertical_convulation(gray, G_y)

"""
cv.imshow('horizontally',anchor1)
cv.waitKey(0)

cv.imshow('vertically',anchor2)
cv.waitKey(0)"""

anchor = convolution.combined(anchor1, anchor2)
print(anchor)
"""from matplotlib import pyplot as plt
plt.imshow(anchor, interpolation='none')
plt.show()"""


data = im.fromarray(anchor)
      
data.save('gfg_dummy_pic.png')

