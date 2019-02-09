import numpy as np
import cv2
import matplotlib.pyplot as plt

nx = 8
ny = 6

img = cv2.imread('./Chessboards/calibration_test.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()