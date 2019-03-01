import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Binarize:

    def __init__(self, image):
        self.image = image
    
    def threshold(image, low, high):
        if len(image.shape) == 2:
            binary = np.zeros_like(image)
            mask = (image >= low) & (image <= high)
        elif len(image.shape) == 3:
            binary = np.zeros_like(image[:, :, 0])
            mask = (image[:, :, 0] >= low[0]) & (image[:, :, 0] <= high[0]) &\
            (image[:, :, 1] >= low[1]) & (image[:, :, 1] <= high[1]) &\
            (image[:, :, 2] >= low[2]) & (image[:, :, 2] <= high[2])
        binary[mask] = 1
        return binary
    
    def binarize_lab(image):
        Binarize.binarize_lab = staticmethod(Binarize.binarize_lab)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        B = lab[:, :, 2]
        L_max, L_mean = np.max(L), np.mean(L)
        B_max, B_mean = np.max(B), np.mean(B)
        L_adapt_yellow = max(80, int(L_max * 0.45))
        B_adapt_yellow =  max(int(B_max * 0.70), int(B_mean * 1.2))
        lab_low_yellow = np.array((L_adapt_yellow, 120, B_adapt_yellow))
        lab_high_yellow = np.array((255, 145, 255))
        lab_yellow = Binarize.threshold(lab, lab_low_yellow, lab_high_yellow)
        lab_binary = lab_yellow
        return lab_binary
    
    def binarize_hsv(image):
        Binarize.binarize_lab = staticmethod(Binarize.binarize_lab)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H = hsv[:,:,0]
        H_max, H_mean = np.max(H), np.mean(H)
        S = hsv[:,:,1]
        S_max, S_mean = np.max(S), np.mean(S)
        V = hsv[:,:,2]
        V_max, V_mean = np.max(V), np.mean(V)        
        # YELLOW
        S_adapt_yellow =  max(int(S_max * 0.25), int(S_mean * 1.75))
        V_adapt_yellow =  max(50, int(V_mean * 1.25))
        hsv_low_yellow = np.array((15, S_adapt_yellow, V_adapt_yellow))
        hsv_high_yellow = np.array((30, 255, 255))
        hsv_yellow = Binarize.threshold(hsv, hsv_low_yellow, hsv_high_yellow)    
        # WHITE
        V_adapt_white = max(150, int(V_max * 0.8),int(V_mean * 1.25))
        hsv_low_white = np.array((0, 0, V_adapt_white))
        hsv_high_white = np.array((255, 40, 220))
        hsv_white = Binarize.threshold(hsv, hsv_low_white, hsv_high_white)
        hsv_binary = hsv_yellow | hsv_white
        return hsv_binary
    
    def binarize_hls(image):
        Binarize.binarize_lab = staticmethod(Binarize.binarize_lab)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        L = hls[:, :, 1]
        L_max, L_mean = np.max(L), np.mean(L)
        S = hls[:, :, 2]
        S_max, S_mean = np.max(S), np.mean(S)
        # YELLOW
        L_adapt_yellow = max(80, int(L_mean * 1.25))
        S_adapt_yellow = max(int(S_max * 0.25), int(S_mean * 1.75))
        hls_low_yellow = np.array((15, L_adapt_yellow, S_adapt_yellow))
        hls_high_yellow = np.array((30, 255, 255))
        hls_yellow = Binarize.threshold(hls, hls_low_yellow, hls_high_yellow)
        # WHITE
        L_adapt_white =  max(160, int(L_max * 0.8),int(L_mean * 1.25))
        hls_low_white = np.array((0, L_adapt_white,  0))
        hls_high_white = np.array((255, 255, 255))
        hls_white = Binarize.threshold(hls, hls_low_white, hls_high_white)
        hls_binary = hls_yellow | hls_white
        return hls_binary
    
    def binarize_red(image):
        Binarize.binarize_lab = staticmethod(Binarize.binarize_lab)
        red = image[:, :, 0]
        red_max, red_mean = np.max(red), np.mean(red)
        red_low_white = min(max(150, int(red_max * 0.55), int(red_mean * 1.95)),230)
        red_binary = Binarize.threshold(red, red_low_white, 255)
        return red_binary
    
    def binarize_adaptive(image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        #Yellow
        adapt_yellow_S = cv2.adaptiveThreshold(hls[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
        adapt_yellow_B = cv2.adaptiveThreshold(lab[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -5)
        adapt_yellow = adapt_yellow_S & adapt_yellow_B
        # White
        adapt_white_R = cv2.adaptiveThreshold(image[:, :, 0], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
        adapt_white_L = cv2.adaptiveThreshold(hsv[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 161, -27)
        adapt_white = adapt_white_R & adapt_white_L
        adapt_binary =  adapt_yellow | adapt_white
        return adapt_binary
    
    def binarize_ensemble(image):
        Binarize.binarize_lab = staticmethod(Binarize.binarize_lab)
        Binarize.binarize_hsv = staticmethod(Binarize.binarize_hsv)
        Binarize.binarize_hls = staticmethod(Binarize.binarize_hls)
        Binarize.binarize_red = staticmethod(Binarize.binarize_red)
        Binarize.binarize_adaptive = staticmethod(Binarize.binarize_adaptive)
        lab_binary = Binarize.binarize_lab(image)
        hsv_binary = Binarize.binarize_hsv(image)
        hls_binary = Binarize.binarize_hls(image)
        red_binary = Binarize.binarize_red(image)
        adaptive_binary = Binarize.binarize_adaptive(image)
        combined = np.asarray(red_binary + lab_binary + hls_binary + hsv_binary + adaptive_binary, dtype = np.uint8)
        combined[combined < 3] = 0
        combined[combined >= 3] = 1
        return combined
    
    def process(self):
        Binarize.binarize_ensemble = staticmethod(Binarize.binarize_ensemble)
        binarized_image = Binarize.binarize_ensemble(self.image)
        return binarized_image