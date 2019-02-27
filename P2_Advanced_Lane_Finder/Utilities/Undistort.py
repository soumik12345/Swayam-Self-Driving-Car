import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Undistort:

    def __init__(self, image, model_path = '../models/', filename = 'camera_callibration.p'):
        self.image = image
        self.model_path = model_path
        self.filename = filename
    
    def is_model_available(model_path):
        if 'camera_callibration.p' in os.listdir(model_path):
            return True
        return False
    
    def load_model(model_path, filename):
        file = open(model_path + filename, 'rb')
        _object = pickle.load(file)
        file.close()
        return _object['mtx'], _object['dist']
    
    def undistort(image, mtx, dist):
        return cv2.undistort(image, mtx, dist, None, mtx)
    
    def generate_callibration_model(shape, model_path):
        images = glob.glob('../Chessboards/8x6/GO*.jpg')
        
        object_points, image_points = [], []
        objp = np.zeros((6 * 8, 3), np.float32)
        objp[:, : 2] = np.mgrid[0 : 8, 0 : 6].T.reshape(-1, 2)
        
        for idx, file_name in enumerate(images):
            img = mpimg.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
            if ret:
                image_points.append(corners)
                object_points.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points,
            (shape[1], shape[0]),
            None, None
            )
        distortion_object = {}
        distortion_object['mtx'] = mtx
        distortion_object['dist'] = dist
        pickle.dump(distortion_object, open(model_path + 'camera_callibration.p', "wb" ))
        return mtx, dist
    
    def process(self):
        Undistort.is_model_available = staticmethod(Undistort.is_model_available)
        Undistort.load_model = staticmethod(Undistort.load_model)
        Undistort.undistort = staticmethod(Undistort.undistort)
        
        if not Undistort.is_model_available(self.model_path):
            Undistort.generate_callibration_model = staticmethod(Undistort.generate_callibration_model)
            mtx, dist = Undistort.generate_callibration_model(self.image.shape, self.model_path)
        else:
            mtx, dist = Undistort.load_model(self.model_path, self.filename)
        
        undistorted = Undistort.undistort(self.image, mtx, dist)
        return undistorted