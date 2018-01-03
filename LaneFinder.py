# Importing Libraries
import numpy as np
from numpy import ones, vstack, zeros_like
from numpy.linalg import lstsq
from statistics import mean
import cv2, pyautogui
from collections import deque

class LaneFinder:
    def __init__(self):
        pass
    
    def process(self, image):
    
        def region_of_interest(img, vertices):
            mask = zeros_like(img) # Empty mask  
            cv2.fillPoly(mask, vertices, 255) # Adding the polygon inside the mask
            return cv2.bitwise_and(img, mask) # Joining the mask and the image
        
        def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):
            try:
                ys = []  # Determining the horizon
                for i in lines:
                    for ii in i:
                        ys += [ii[1],ii[3]]
                min_y = min(ys)
                max_y = 600
                new_lines = []
                line_dict = {}
                for idx,i in enumerate(lines):
                    for cord in i:
                        x_coords = (cord[0], cord[2])
                        y_coords = (cord[1], cord[3])
                        A = vstack([x_coords,ones(len(x_coords))]).T
                        m, b = lstsq(A, y_coords)[0]
                        x1, x2 = 0, 0
                        if m!=0:
                            x1 = (min_y-b) / m
                            x2 = (max_y-b) / m
                        else: # Avoiding division by zero
                            x1, x2 = 9223372036854775807, 9223372036854775807
                        line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                        new_lines.append([int(x1), min_y, int(x2), max_y])
                final_lanes = {}
                for idx in line_dict:
                    final_lanes_copy = final_lanes.copy()
                    m = line_dict[idx][0]
                    b = line_dict[idx][1]
                    line = line_dict[idx][2]
                    if len(final_lanes) == 0:
                        final_lanes[m] = [ [m,b,line] ]
                    else:
                        found_copy = False
                        for other_ms in final_lanes_copy:
                            if not found_copy:
                                if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                                    if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                        final_lanes[other_ms].append([m,b,line])
                                        found_copy = True
                                        break
                                else:
                                    final_lanes[m] = [ [m,b,line] ]
                line_counter = {}
                for lanes in final_lanes:
                    line_counter[lanes] = len(final_lanes[lanes])
                top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
                lane1_id = top_lanes[0][0]
                lane2_id = top_lanes[1][0]
                def average_lane(lane_data):
                    x1s, y1s, x2s, y2s = [], [], [], []
                    for data in lane_data:
                        x1s.append(data[2][0])
                        y1s.append(data[2][1])
                        x2s.append(data[2][2])
                        y2s.append(data[2][3])
                    return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 
                l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
                l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
                return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
            except Exception as e:
                print(str(e))
                pass
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.Canny(image, threshold1 = 200, threshold2=300)
        image = cv2.GaussianBlur(image,(5,5),0)
        vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]], np.int32)
        image = region_of_interest(image, [vertices])
        lines = cv2.HoughLinesP(image, 1, np.pi/180, 180,      20,       15)
        try:
            l1, l2 = draw_lanes(original_image,lines)
            cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
            cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
        except Exception as e:
            pass
        try:
            for coords in lines:
                coords = coords[0]
                try:
                    cv2.line(image, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                except Exception as e:
                    print(str(e))
        except Exception as e:
            pass
        return original_image
        
