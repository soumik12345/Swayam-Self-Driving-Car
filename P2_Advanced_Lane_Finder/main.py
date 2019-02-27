import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Utilities.Undistort import *

undistort = Undistort(
    mpimg.imread('./Chessboards/8x6/test_image.jpg'),
    model_path = './models/'
)
undistorted = undistort.process()
plt.imshow(undistorted)
plt.show()