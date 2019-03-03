import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Utilities.Undistort import *

class Perspective_Transform:

    def __init__(self, image):
        self.image = image
    
    def get_roi(image, vertices):
        vertices = np.array(vertices, ndmin = 3, dtype = np.int32)
        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, vertices, (255, 255, 255))
        return cv2.bitwise_and(image, mask)
    
    def warp_image(image, warp_shape, source, destination):
        M = cv2.getPerspectiveTransform(source, destination)
        inverse_M = cv2.getPerspectiveTransform(destination, source)
        warped = cv2.warpPerspective(image, M, warp_shape, flags = cv2.INTER_LINEAR)
        return warped, M, inverse_M
    
    def perspective_transform(image):
        y, x, _ = image.shape
        source = np.float32([
            (696, 455),    
            (587, 455), 
            (235, 700),  
            (1075, 700)
        ])
        destination = np.float32([
            (x - 350, 0),
            (350, 0),
            (350, y),
            (x - 350, y)
        ])
        image_labelled = image.copy()
        cv2.polylines(image_labelled, [np.int32(source)], True, (255, 0, 0), 3)
        warped, M, invM = Perspective_Transform.warp_image(image, (x, y), source, destination)
        warped_labelled = warped.copy()
        cv2.polylines(warped_labelled, [np.int32(destination)], True, (255, 0, 0), 3)
        vertices = np.array([
            [200, y],
            [200, 0],
            [1100, 0],
            [1100, y]
        ])
        roi = Perspective_Transform.get_roi(warped, vertices)
        return warped, warped_labelled, roi, image_labelled, M, invM
    
    def process(self):
        Perspective_Transform.get_roi = staticmethod(Perspective_Transform.get_roi)
        Perspective_Transform.warp_image = staticmethod(Perspective_Transform.warp_image)
        Perspective_Transform.perspective_transform = staticmethod(Perspective_Transform.perspective_transform)

        warped, warped_labelled, roi, image_labelled, M, invM = Perspective_Transform.perspective_transform(self.image)

        return warped, warped_labelled, roi, image_labelled, M, invM