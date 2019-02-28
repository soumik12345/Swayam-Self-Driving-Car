import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Utilities.Undistort import *
from Utilities.Perspective_Transform import *

undistort = Undistort(
    mpimg.imread('./Lane Images/straight_lines2.jpg'),
    model_path = './models/'
)
undistorted_image = undistort.process()
plt.imshow(undistorted_image)
plt.title('Undistorted')
plt.show()

perspective_transform = Perspective_Transform(undistorted_image)
warped, warped_labelled, roi, image_labelled = perspective_transform.process()
plt.imshow(image_labelled)
plt.title('Labelled Image')
plt.show()
plt.imshow(warped_labelled)
plt.title('Birds Eye View')
plt.show()