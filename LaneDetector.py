# Importing Libraries
import numpy as np
import cv2, pyautogui
from collections import deque

class LaneDetector:
    def __init__(self):
        self.left_lines = deque(maxlen = 50)
        self.right_lines = deque(maxlen = 50)
    
    def process(self, image):
        
        def convert_hsv(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        def convert_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        def convert_hsl(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        
        def convert_gray(img):
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        def smoothen(img, kernel_size=15):
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Selecting white and yellow colors from RGB
        def select_rgb_white_yellow(img):
            # White mask
            lower, upper = np.uint8([200, 200, 200]), np.uint8([255, 255, 255])
            white = cv2.inRange(img, lower, upper)
            # Yellow mask
            lower, upper = np.uint8([190, 190, 0]), np.uint8([255, 255, 255])
            yellow = cv2.inRange(img, lower, upper)
            mask = cv2.bitwise_or(white, yellow) # Combining the masks
            mask_img = cv2.bitwise_and(img, img, mask = mask)
            return mask_img
        
        # Selecting white and yellow colors from HSL
        def select_hsl_white_yellow(img):
            img=convert_hsl(img)
            # White mask
            lower, upper = np.uint8([0, 200, 0]), np.uint8([255, 255, 255])
            white = cv2.inRange(img, lower, upper)
            # Yellow mask
            lower, upper = np.uint8([10, 0, 100]), np.uint8([40, 255, 255])
            yellow = cv2.inRange(img, lower, upper)
            mask = cv2.bitwise_or(white, yellow) # Combining the masks
            mask_img = cv2.bitwise_and(img, img, mask = mask)
            return mask_img
        
        # Detecting edges
        def detect_edges(img, l_threshold = 50, u_threshold = 150):
            return cv2.Canny(img, l_threshold, u_threshold)
        
        def filter_region(img, vertices):
            mask = np.zeros_like(image)
            if len(mask.shape) == 2:
                cv2.fillPoly(mask, vertices, 255)
            else:
                cv2.fillPoly(mask, vertices, (255, ) * mask.shape[2])
            return cv2.bitwise_and(image, mask)
        
        def region_of_interest(img):
            vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
            return filter_region(img, [vertices])
        
        def hough_lines(img):
            # return cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
            return cv2.HoughLinesP(img, 1, np.pi/180, 180, 20, 15)
        
        def draw_lines(img, lines, color=[0, 255, 0], thickness=2):
            for i in lines:
                for x1, y1, x2, y2 in i:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            return img
        
        def average_slope_intercept(lines):
            left_lines, left_weights = [], []
            right_lines, right_weights = [], []
            for i in lines:
                for x1, y1, x2, y2 in i:
                    if x1 == x2: # Vertical lines are ignores
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y1 - slope * x1
                    length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2) # Euclidean Distance
                    if slope < 0:
                        left_lines.append((slope, intercept))
                        left_weights.append((length))
                    else:
                        right_lines.append((slope, intercept))
                        right_weights.append((length))
            left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
            right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
            return left_lane, right_lane
        
        def make_line_points(y1, y2, line):
            if line is None:
                return None
            slope, intercept = line
            x1 = int((y1 - intercept) / slope) 
            x2 = int((y2 - intercept) / slope)
            y1, y2 = int(y1), int(y2)
            return ((x1, y1), (x2, y2))
        
        def lane_lines(img, lines):
            left_lane, right_lane = average_slope_intercept(lines)
            y1 = img.shape[0] # image bottom
            y2 = 0.6 * y1 # middle
            left_line, right_line = make_line_points(y1, y2, left_lane), make_line_points(y1, y2, right_lane)
            return left_line, right_line
        
        def draw_lane_lines(img, lines, color = [255, 0, 0], thickness = 20):
            line_image = np.zeros_like(img)
            for i in lines:
                if i is not None:
                    cv2.line(line_image, *i, color, thickness)
            return cv2.addWeighted(img, 1.0, line_image, 0.95, 0.0)
        
        white_yellow = select_hsl_white_yellow(image)
        gray = convert_gray(white_yellow)
        smooth = smoothen(gray)
        edges = detect_edges(smooth)
        roi = region_of_interest(edges)
        lines = hough_lines(edges)
        left_line, right_line = lane_lines(image, lines)

        def mean_line(line, lines):
            if line is not None:
                lines.append(line)
            if len(lines) > 0:
                line = np.mean(lines, axis = 0, dtype = np.int32)
                line = tuple(map(tuple, line))
            return line
        
        left_line = mean_line(left_line, self.left_lines)
        right_line = mean_line(right_line, self.right_lines)

        return draw_lane_lines(image, (left_line, right_line))