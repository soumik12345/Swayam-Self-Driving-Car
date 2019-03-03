import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2, glob, os, pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Lane_Detector:
    
    def __init__(self, image, binary_image, camera_matrix, camera_matrix_inverse):
        self.image = image
        self.binary_image = binary_image
        self.shape = binary_image.shape
        self.camera_matrix = camera_matrix
        self.camera_matrix_inverse = camera_matrix_inverse
    
    def get_histogram(image):
        return np.sum(image[image.shape[0] // 2 :, :], axis = 0)
    
    def get_histogram_peaks(histogram):
        middle = np.int(histogram.shape[0] / 2)
        left_peak = np.argmax(histogram[ : middle])
        right_peak = np.argmax(histogram[middle : ]) + middle
        return left_peak, right_peak
    
    def get_window_boundaries(image, window, window_height, margin, left_x, right_x):
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_x_left_low = left_x - margin
        win_x_left_high = left_x + margin
        win_x_right_low = right_x - margin
        win_x_right_high = right_x + margin
        return win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high
    
    def detect_lane_pixels(binary_image, n_windows = 9, margin = 100, min_pix = 50):
        Lane_Detector.get_histogram = staticmethod(Lane_Detector.get_histogram)
        Lane_Detector.get_histogram_peaks = staticmethod(Lane_Detector.get_histogram_peaks)
        Lane_Detector.get_window_boundaries = staticmethod(Lane_Detector.get_window_boundaries)

        histogram = Lane_Detector.get_histogram(binary_image)
        left_x_base, right_x_base = Lane_Detector.get_histogram_peaks(histogram)
        output_image = np.dstack((binary_image, binary_image, binary_image)) * 255
        window_height = np.int(binary_image.shape[0] // n_windows)
        nonzero_y = np.array(binary_image.nonzero()[0])
        nonzero_x = np.array(binary_image.nonzero()[1])
        left_x_current = left_x_base
        right_x_current = right_x_base
        left_lane_indices, right_lane_indices = [], []
        for window in range(n_windows):
            boundaries = Lane_Detector.get_window_boundaries(binary_image, window, window_height, margin, left_x_current, right_x_current)
            win_y_low, win_y_high, win_x_left_low, win_x_left_high, win_x_right_low, win_x_right_high = boundaries
            cv2.rectangle(output_image, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2) 
            cv2.rectangle(output_image, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
            good_left_indices = ((nonzero_y >= win_y_low) &\
                                (nonzero_y < win_y_high) &\
                                (nonzero_x >= win_x_left_low) &\
                                (nonzero_x < win_x_left_high)).nonzero()[0]
            good_right_indices = ((nonzero_y >= win_y_low) &\
                                (nonzero_y < win_y_high) &\
                                (nonzero_x >= win_x_right_low) &\
                                (nonzero_x < win_x_right_high)).nonzero()[0]
            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)
            if len(good_left_indices) > min_pix:
                left_x_current = np.int(np.mean(nonzero_x[good_left_indices]))
            if len(good_right_indices) > min_pix:        
                right_x_current = np.int(np.mean(nonzero_x[good_right_indices]))
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
        left_x = nonzero_x[left_lane_indices]
        left_y = nonzero_y[left_lane_indices] 
        right_x = nonzero_x[right_lane_indices]
        right_y = nonzero_y[right_lane_indices]
        output_image[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
        output_image[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]
        return output_image, left_x, left_y, left_lane_indices, right_x, right_y, right_lane_indices, nonzero_x, nonzero_y
    
    def fit_polynomial(binary_image, left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)
        plot_y = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
        left_fit_x = left_fit[0] * (plot_y ** 2) + left_fit[1] * plot_y + left_fit[2]
        right_fit_x = right_fit[0] * (plot_y ** 2) + right_fit[1] * plot_y + right_fit[2]
        return plot_y, left_fit, left_fit_x, right_fit, right_fit_x
    
    def detect_lane_lines(binary_image, n_windows = 9, margin = 100, min_pix = 50):
        Lane_Detector.detect_lane_pixels = staticmethod(Lane_Detector.detect_lane_pixels)
        Lane_Detector.fit_polynomial = staticmethod(Lane_Detector.fit_polynomial)

        sliding_window_image, left_x, left_y, left_lane_indices, right_x, right_y, right_lane_indices, nonzero_x, nonzero_y = Lane_Detector.detect_lane_pixels(binary_image, n_windows, margin, min_pix)
        plot_y, left_fit, left_fit_x, right_fit, right_fit_x = Lane_Detector.fit_polynomial(binary_image, left_x, left_y, right_x, right_y)
        output_image_1 = np.dstack((binary_image, binary_image, binary_image)) * 255
        output_image_2 = np.zeros_like(output_image_1)
        output_image_1[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
        output_image_1[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]
        
        left_line_window_1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
        left_line_points = np.hstack((left_line_window_1, left_line_window_2))
        
        right_line_window_1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window_2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
        right_line_points = np.hstack((right_line_window_1, right_line_window_2))
        
        cv2.fillPoly(output_image_2, np.int_([left_line_points]), (0, 255, 0))
        cv2.fillPoly(output_image_2, np.int_([right_line_points]), (0, 255, 0))
        result = cv2.addWeighted(output_image_1, 1, output_image_2, 0.3, 0)
        
        left_curve_radius = ((1 + (2 * left_fit[0] * np.max(plot_y) + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curve_radius = ((1 + (2 * right_fit[0] * np.max(plot_y) + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        
        return sliding_window_image, result, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius
    
    def unwarp_image(image, binary_image, M, Minv, margin = 100):
        Lane_Detector.detect_lane_lines = staticmethod(Lane_Detector.detect_lane_lines)
        sliding_window_image, result, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius = Lane_Detector.detect_lane_lines(binary_image, margin = margin)
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
        points_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        points_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        points = np.hstack((points_left, points_right))
        color_warp = np.dstack((np.zeros_like(binary_image).astype(np.uint8),
                                np.zeros_like(binary_image).astype(np.uint8),
                                np.zeros_like(binary_image).astype(np.uint8)))
        cv2.polylines(color_warp, np.int_([points]), isClosed = False, color = (0, 0, 255), thickness = 40)
        cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (binary_image.shape[1], binary_image.shape[0]))
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result

    def process(self):
        Lane_Detector.detect_lane_lines = staticmethod(Lane_Detector.detect_lane_lines)
        Lane_Detector.unwarp_image = staticmethod(Lane_Detector.unwarp_image)

        sliding_window_image, result, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius = Lane_Detector.detect_lane_lines(self.binary_image)
        final_image = Lane_Detector.unwarp_image(self.image, self.binary_image, self.camera_matrix, self.camera_matrix_inverse)
        return sliding_window_image, result, final_image, left_fit_x, right_fit_x, plot_y, left_curve_radius, right_curve_radius