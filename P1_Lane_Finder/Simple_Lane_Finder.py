import numpy as np
import cv2

class Simple_Lane_Finder(object):
    
    def __init__(self, image):
        self.image = image
    
    def process(self):

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
            filtered_image = cv2.bitwise_and(image, image, mask = mask)
            return filtered_image
        
        def convert_to_gray(image):
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        def apply_gaussian_blur(image, k_size = 15):
            return cv2.GaussianBlur(image, (k_size, k_size), 0)
        
        def edge_detection(image, low_threshold = 50, high_threshold = 150):
            return cv2.Canny(image, low_threshold, high_threshold)
        
        def select_mask(image, vertices):
            mask = np.zeros_like(image) # Empty copy of the image
            if len(mask.shape)==2:
                cv2.fillPoly(mask, vertices, 255)
            else: # in case, the input image has a channel dimension
                cv2.fillPoly(mask, vertices, (255,)*mask.shape[2])        
            return cv2.bitwise_and(image, mask)
        
        def select_region(image):
            rows, cols = image.shape[:2]
            bottom_left = [cols*0.1, rows*0.95]
            top_left = [cols*0.4, rows*0.6]
            bottom_right = [cols*0.9, rows*0.95]
            top_right = [cols*0.6, rows*0.6] 
            # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
            vertices = np.array([[
                bottom_left, top_left,
                top_right,
                bottom_right
            ]], dtype = np.int32)
            return select_mask(image, vertices)
        
        def hough_line_transforms(image):
            return cv2.HoughLinesP(
                image, rho = 1,
                theta = np.pi / 180,
                threshold = 20,
                minLineLength = 20,
                maxLineGap = 300
            )
        
        def draw_lines(image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
            # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
            if make_copy:
                image = np.copy(image) # making a copy of the original
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
            return image
        
        def average_slope_intercept(lines):
            left_lines, right_lines = [], [] # (slope, intercept)
            left_weights, right_weights = [], [] # (length)
            
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 == x2: # ingnore a vertical line
                        continue
                    m = (y2 - y1) / (x2 - x1) # slope
                    c = y1 - m * x1 # y-intercept
                    dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) # euclidean distance
                    
                    if m < 0: # slope is negative
                        left_lines.append((m, c))
                        left_weights.append((dist))
                    else:
                        right_lines.append((m, c))
                        right_weights.append((dist))
            
            # adding weightage to longer lines
            left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
            right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
            
            return left_lane, right_lane
        
        def make_line_coords(y1, y2, line):
            if line is None:
                return None
            m, c = line
            x1, x2 = int((y1 - c) / m), int((y2 - c) / m)
            y1, y2 = int(y1), int(y2)
            return ((x1, y1), (x2, y2))
        
        def make_line_coords(y1, y2, line):
            if line is None:
                return None
            m, c = line
            x1, x2 = int((y1 - c) / m), int((y2 - c) / m)
            y1, y2 = int(y1), int(y2)
            return ((x1, y1), (x2, y2))

        def lane_lines(image, lines):
            left_lane, right_lane = average_slope_intercept(lines)
            y1 = image.shape[0] # bottom of the image
            y2 = y1 * 0.6 # middle (a bit lower)
            left_line  = make_line_coords(y1, y2, left_lane)
            right_line = make_line_coords(y1, y2, right_lane)
            return left_line, right_line
        
        def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
            # make a separate image to draw lines and combine with the orignal later
            line_image = np.zeros_like(image)
            for line in lines:
                if line is not None:
                    cv2.line(line_image, *line,  color, thickness)
            # image1 * α + image2 * β + λ
            # image1 and image2 must be the same shape.
            return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

        hls_white_yellow_image = hsl_white_yellow(self.image)
        gray = convert_to_gray(hls_white_yellow_image)
        blur = apply_gaussian_blur(gray)
        edge = edge_detection(blur)
        roi = select_region(edge)
        lines = hough_line_transforms(roi)
        try:
            line_image = draw_lines(self.image, lines)
        except TypeError:
            line_image = self.image
        try:
            result = draw_lane_lines(line_image, lane_lines(line_image, lines))
        except:
            result = self.image

        return result