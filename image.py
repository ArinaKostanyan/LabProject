import cv2 as cv
import gaussian
import filtering
import numpy as np

def processing(path, kernel, gaussian_kernel):
    if kernel is None:
        kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
    
    # Тhis method loads an image from the specified file
    img = cv.imread(path)

    # Converts an image from one color space to another
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Gaussian blurring
    gaus = gaussian.blurring(gray, gaussian_kernel)

    anchor1 = filtering.convolution(gaus, kernel)
    anchor2 = filtering.convolution(gaus, kernel.transpose())
    anchor = filtering.combinacia(anchor1, anchor2)
    
    # Vertical and horizontal convolution showing
    filename = 'anchor1.jpeg'
    cv.imshow(filename, anchor1)
    cv.imwrite(filename, anchor1)

    filename = 'anchor2.jpeg'
    cv.imshow(filename, anchor2)
    cv.imwrite(filename, anchor2)

    # Edge detected image showing
    filename = 'Edge_detected.jpeg'
    cv.imshow(filename, anchor)
    cv.imwrite(filename, anchor)

    cv.waitKey(7000)
    cv.destroyAllWindows()
