import cv2 as cv
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def findClickPositions(needle_img_path, haystack_img_path, threshold=0.5, debug_mode=None):

    obj_img = cv.imread(needle_img_path, cv.IMREAD_UNCHANGED)
    orb_img = cv.imread(haystack_img_path, cv.IMREAD_UNCHANGED)

    orb_w = orb_img.shape[1]
    orb_h = orb_img.shape[0]

    method = cv.TM_SQDIFF_NORMED
    result = cv.matchTemplate(obj_img, orb_img, method)

    locations =  np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))

    rectangles = []

    for loc in locations:
        rect = [int(loc[0]), int(loc[1]), orb_w, orb_h]
        rectangles.append(rect)
        rectangles.append(rect)

    rectangles, weights = cv.groupRectangles(rectangles, 1, 0.5)

    #print(rectangles)
    points = []
    if len(rectangles): 
        #print('Found Orb.')


        line_color = (0,255,0)
        line_type = cv.LINE_4
        maker_color = (255, 0, 255)
        marker_type = cv.MARKER_CROSS

        for (x, y, w, h) in rectangles:
            center_x = x + int(w/2)
            center_y = y + int(h/2)
            points.append((center_x, center_y))

            if debug_mode == 'rectangles':
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv.rectangle(obj_img, top_left, bottom_right, line_color, line_type)
            elif debug_mode== 'poinst':
                cv.drawMarker(obj_img, (center_x, center_y), maker_color, marker_type)

    if debug_mode:
        cv.imshow('Result', obj_img)
        cv.waitKey()
        #cv.imwrite('result.jpg', obj_img)
            
    return 