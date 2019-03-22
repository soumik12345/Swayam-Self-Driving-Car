import warnings
warnings.filterwarnings('ignore')
import cv2, glob
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class Car_Classifier:

    def __init__(self):
        self.cars = glob.glob('../Data/vehicles_smallset/cars1/*.jpeg') + glob.glob('../Data/vehicles_smallset/cars2/*.jpeg') + glob.glob('../Data/vehicles_smallset/cars3/*.jpeg')
        self.non_cars = glob.glob('../Data/non-vehicles_smallset/notcars1/*.jpeg') + glob.glob('../Data/non-vehicles_smallset/notcars2/*.jpeg') + glob.glob('../Data/non-vehicles_smallset/notcars3/*.jpeg')
    
    def get_hog_features(image, orientations, pixels_per_cell, cells_per_block):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features, hog_image = hog(
            image, orientations = orientations,
            pixels_per_cell = (pixels_per_cell, pixels_per_cell),
            cells_per_block = (cells_per_block, cells_per_block),
            visualise = True, feature_vector = False,
            block_norm = "L2-Hys"
        )
        return hog_features, hog_image
    
    def get_features(image, colorspace = 'RGB', orientations = 9, pixels_per_cell = 8, cells_per_block = 2):
        if colorspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif colorspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif colorspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif colorspace == 'RGB' or colorspace == 'BGR':
            feature_image = np.copy(image)
        Car_Classifier.get_hog_features = staticmethod(Car_Classifier.get_hog_features)
        features, _ = Car_Classifier.get_hog_features(
            feature_image,
            orientations,
            pixels_per_cell,
            cells_per_block
        )
        flattened_features = np.ravel(features)
        return flattened_features
    
    def preprocess_dataset(cars, non_cars):
        Car_Classifier.get_features = staticmethod(Car_Classifier.get_features)
        features_cars = [Car_Classifier.get_features(mpimg.imread(image_location)) for image_location in cars]
        features_not_cars = [Car_Classifier.get_features(mpimg.imread(image_location)) for image_location in non_cars]
        label_cars = [1] * len(cars)
        label_not_cars = [0] * len(non_cars)
        x = features_cars + features_not_cars
        x = np.vstack(x).astype(np.float64)
        y = label_cars + label_not_cars
        y = np.vstack(y).astype(np.float64)
        x_scaler = StandardScaler().fit(x)
        x_scaled = x_scaler.transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 100)
        return x_train, y_train, x_scaler
    
    def train_svm(x_train, y_train):
        print('Training Support Vector Machine....')
        svc = LinearSVC()
        svc.fit(x_train, y_train)
        print('Done.')
        return svc
    
    def get_models(self):
        Car_Classifier.preprocess_dataset = staticmethod(Car_Classifier.preprocess_dataset)
        Car_Classifier.train_svm = staticmethod(Car_Classifier.train_svm)
        print('Preprocessing Dataset....')
        x_train, y_train, x_scaler = Car_Classifier.preprocess_dataset(self.cars, self.non_cars)
        print('Done.')
        classifier = Car_Classifier.train_svm(x_train, y_train)
        return x_scaler, classifier