import cv2
import numpy as np
import copy
import math

import settings
import detection

blur_kernel_min = 1
blur_kernel_max = 51

canny_lowThreshold = 0
canny_max_lowThreshold = 100

# canny_ratio = 3
# canny_kernel_size = 3

line_threshold_min = 10
line_threshold_max = 100
line_minLineLength_min = 10
line_minLineLength_max = 200
line_maxLineGap_min = 1
line_maxLineGap_max = 20

original_img = cv2.imread('F:/meteor-one-click-test-original-images/star-aligned/IMG_3800_r.jpg')
next_img = cv2.imread('F:/meteor-one-click-test-original-images/star-aligned/IMG_3801_r.jpg')

meteor_detector = detection.MeteorDetector()


def enhance_img_by_subtraction(original_img, next_img):
    img = cv2.subtract(original_img, next_img)
    return img

enhanced_img = enhance_img_by_subtraction(original_img, next_img)


def Blur_Canny_HoughLinesP_on_draw(x):
    detection_lines = meteor_detector.detect_meteor_from_image(enhanced_img, original_img)
    if not (detection_lines is None):
        if len(detection_lines) < 200:
            draw_img = meteor_detector.draw_detection_boxes_on_image(original_img, detection_lines, color=(255,255,0))

            draw_img = cv2.resize(draw_img, (1280, 853))
            img = cv2.resize(enhanced_img, (1280, 853))

            # detected_edges_enh = cv2.resize(detected_edges_enh, (1280, 853))
            # img_enh = cv2.resize(img_enh, (1280, 853))

            cv2.imshow('Meteor edge', draw_img)
            cv2.imshow('Meteor detect', img)


if __name__ == "__main__":


    cv2.namedWindow('Meteor detection', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Meteor detection', cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('Blur kernel size', 'Meteor detection',
                       blur_kernel_min,
                       blur_kernel_max,
                       Blur_Canny_HoughLinesP_on_draw)
    cv2.setTrackbarPos('Blur kernel size', 'Meteor detection', 3)

    cv2.createTrackbar('Canny min threshold', 'Meteor detection',
                       canny_lowThreshold,
                       canny_max_lowThreshold,
                       Blur_Canny_HoughLinesP_on_draw)
    cv2.setTrackbarPos('Canny min threshold', 'Meteor detection', 40)

    cv2.createTrackbar('Line threshold', 'Meteor detection',
                       line_threshold_min,
                       line_threshold_max,
                       Blur_Canny_HoughLinesP_on_draw)
    cv2.setTrackbarPos('Line threshold', 'Meteor detection', 50)

    cv2.createTrackbar('Min line length', 'Meteor detection',
                       line_minLineLength_min,
                       line_minLineLength_max,
                       Blur_Canny_HoughLinesP_on_draw)
    cv2.setTrackbarPos('Min line length', 'Meteor detection', 30)

    cv2.createTrackbar('Max line gap', 'Meteor detection',
                       line_maxLineGap_min,
                       line_maxLineGap_max,
                       Blur_Canny_HoughLinesP_on_draw)
    cv2.setTrackbarPos('Max line gap', 'Meteor detection', 3)

    Blur_Canny_HoughLinesP_on_draw(0)  # initialization

    cv2.namedWindow("Meteor edge", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Meteor detect", cv2.WINDOW_AUTOSIZE)

    '''
    cv2.namedWindow("Meteor edge - enhanced", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Meteor detect  - enhanced", cv2.WINDOW_AUTOSIZE)
    '''

    # k = cv2.waitKey(50) & 0xFF
    # if k == 27:
    #     break

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()