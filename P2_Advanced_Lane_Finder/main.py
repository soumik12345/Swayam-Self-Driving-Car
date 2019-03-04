import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Utilities.Perspective_Transform import *
from Utilities.Lane_Detector import *
os.system('cls')
print('Libraries Imported')

def load_model(file):
    file = open(file, 'rb')
    _object = pickle.load(file)
    file.close()
    return _object['mtx'], _object['dist']

def hsl_white_yellow(image):
    # convert to hls
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white = cv2.inRange(hsl, lower, upper)
    # yellow mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow = cv2.inRange(hsl, lower, upper)
    # Combining the mask and the image
    mask = cv2.bitwise_or(white, yellow)
    return mask

def pipeline(frame, mtx, dist):
    undistorted_image = cv2.undistort(frame, mtx, dist, None, mtx)
    perspective_transformer = Perspective_Transform(undistorted_image)
    warped, warped_labelled, roi, image_labelled, M, invM = perspective_transformer.process()
    binary_image = hsl_white_yellow(warped)
    lane_detector = Lane_Detector(frame, binary_image, M, invM)
    sliding_window_image, result, final_image, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius = lane_detector.process()
    return final_image, left_curve_radius, right_curve_radius

def main():
    mtx, dist = load_model('./models/camera_callibration.p')
    print('Camera Model Loaded')
    if sys.argv[1] == 'V':
        cap = cv2.VideoCapture(sys.argv[2])
        out = cv2.VideoWriter(sys.argv[3], -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        print('capture created')
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                final_image, left_curve_radius, right_curve_radius = pipeline(rgb_frame, mtx, dist)
                final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                cv2.putText(final_image, 'Left Curve Radius:' + str(left_curve_radius), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv2.putText(final_image, 'Right Curve Radius:' + str(right_curve_radius), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                out.write(final_image)
            except:
                break
            cv2.imshow(sys.argv[2] + '_final_output', final_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break

main()