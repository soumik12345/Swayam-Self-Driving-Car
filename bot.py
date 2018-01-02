# Importing the libraries
import numpy as np
from PIL import ImageGrab
import cv2, pyautogui, LaneFinder
from directkeys import ReleaseKey, PressKey, W, A, S, D
from collections import deque

def grab_frame():
    return np.array(ImageGrab.grab(bbox = (0,40, 800, 640)))

def process_img(image):
    detector = LaneFinder.LaneFinder()
    img = detector.process(image)
    return img

def main():
    while(True):
        screen = process_img(grab_frame())
        try:
            cv2.imshow('window', screen)
            #cv2.imshow('window2', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        except:
            print('ERROR')

main()