import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Utilities.Undistort import *
from Utilities.Perspective_Transform import *
from Utilities.Binarize import *
from Utilities.Lane_Detector import *

undistort = Undistort(
    mpimg.imread('./Lane Images/straight_lines2.jpg'),
    model_path = './models/'
)
undistorted_image = undistort.process()
plt.imshow(undistorted_image)
plt.title('Undistorted')
plt.show()

perspective_transform = Perspective_Transform(undistorted_image)
warped, warped_labelled, roi, image_labelled, M, invM = perspective_transform.process()
plt.imshow(image_labelled)
plt.title('Labelled Image')
plt.show()
plt.imshow(warped_labelled)
plt.title('Birds Eye View')
plt.show()

binarize = Binarize(warped)
binarized_image = binarize.process()
plt.imshow(binarized_image, cmap = 'gray')
plt.title('Binarized Image')
plt.show()

lane_detector = Lane_Detector(undistorted_image, binarized_image, M, invM)
sliding_window_image, result, final_image, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius = lane_detector.process()
plt.imshow(final_image)
plt.title('Lane Detection')
plt.xlabel('Left Radius Of Curvature: ' + str(left_curve_radius) + '\nRight Radius Of Curvature: ' + str(right_curve_radius))
plt.show()