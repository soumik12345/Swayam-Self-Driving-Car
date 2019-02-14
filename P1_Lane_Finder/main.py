import numpy as np
from PIL import ImageGrab
import cv2, os, sys
from Simple_Lane_Finder import Simple_Lane_Finder

def grab_frame(box_coords):
    return np.array(ImageGrab.grab(bbox = box_coords))

def process(frame):
    simple_lane_finder = Simple_Lane_Finder(frame)
    return simple_lane_finder.process()

def main():
    if sys.argv[1] == 'S':
        while True:
            frame = process(grab_frame((0,40, 800, 640)))
            try:
                cv2.imshow('Simulator', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except:
                pass
    elif sys.argv[1] == 'V':
        cap = cv2.VideoCapture(sys.argv[2])
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                frame = process(frame)
            except:
                break
            try:
                cv2.imshow(sys.argv[2], frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            except:
                pass

main()