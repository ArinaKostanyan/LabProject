import cv2 as cv

import convolution
import gaussian

path = input('Enter your path or default will be picture of Lena \n')

# Тhis method loads an image from the specified file
img = cv.imread(path)
    
if img == None or img == '':
    path = 'image/lena.png'
    img = cv.imread(path)

# Converts an image from one color space to another
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Gaussian blurring
Gaus = gaussian.blurring(gray)

anchor1 = convolution.horizontal_convulation(gray)
anchor2 = convolution.vertical_convulation(gray)
anchor = convolution.combined(gray)

# Printing Gaussian blurred image 
# filename = 'Gaus.jpeg'
# cv.imshow(filename, Gaus)
# cv.imwrite(filename, Gaus)


# Vertical and horizontal convolution printing
filename = 'convertScaleAbs(anchor1)_gaus.jpeg'
cv.imshow(filename, anchor1)
cv.imwrite(filename, anchor1)

filename = 'convertScaleAbs(anchor2)_gaus.jpeg'
cv.imshow(filename, anchor2)
cv.imwrite(filename, anchor2)

# Combined edge detected image printing
filename = 'Edge_detected.jpeg'
cv.imshow(filename, anchor)
cv.imwrite(filename, anchor)

cv.waitKey(0)
cv.destroyAllWindows()