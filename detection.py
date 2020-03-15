# -*- coding: utf-8 -*-
import cv2
import numpy as np
import copy
import math
import os

import settings


class HoughBundler:
    '''
    source:
    https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp

    Clasterize and merge each cluster of cv2.HoughLinesP() output
    a = HoughBundler()
    foo = a.process_lines(houghP_lines, binary_image)
    '''

    def get_orientation(self, line):
        '''get orientation of a line, using its length
        https://en.wikipedia.org/wiki/Atan2
        '''
        orientation = math.atan2(abs((line[0] - line[2])), abs((line[1] - line[3])))
        return math.degrees(orientation)

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        '''Check if line have enough distance and angle to be count as similar
        '''
        for group in groups:
            # walk through existing line groups
            for line_old in group:
                # check distance
                if self.get_distance(line_old, line_new) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_new = self.get_orientation(line_new)
                    orientation_old = self.get_orientation(line_old)
                    # if all is ok -- line is similar to others in group
                    if abs(orientation_new - orientation_old) < min_angle_to_merge:
                        group.append(line_new)
                        return False
        # if it is totally different line
        return True

    def DistancePointLine(self, point, line):
        """Get distance between point and line
        http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
        """
        px, py = point
        x1, y1, x2, y2 = line

        def lineMagnitude(x1, y1, x2, y2):
            'Get line (aka vector) length'
            lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return lineMagnitude

        LineMag = lineMagnitude(x1, y1, x2, y2)
        if LineMag < 0.00000001:
            DistancePointLine = 9999
            return DistancePointLine

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (LineMag * LineMag)

        if (u < 0.00001) or (u > 1):
            # // closest point does not fall within the line segment, take the shorter distance
            # // to an endpoint
            ix = lineMagnitude(px, py, x1, y1)
            iy = lineMagnitude(px, py, x2, y2)
            if ix > iy:
                DistancePointLine = iy
            else:
                DistancePointLine = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            DistancePointLine = lineMagnitude(px, py, ix, iy)

        return DistancePointLine

    def get_distance(self, a_line, b_line):
        """Get all possible distances between each dot of two lines and second line
        return the shortest
        """
        dist1 = self.DistancePointLine(a_line[:2], b_line)
        dist2 = self.DistancePointLine(a_line[2:], b_line)
        dist3 = self.DistancePointLine(b_line[:2], a_line)
        dist4 = self.DistancePointLine(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_pipeline_2(self, lines):
        'Clusterize (group) lines'
        groups = []  # all lines groups are here
        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        return groups

    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and last coordinates
        """
        orientation = self.get_orientation(lines[0])

        # special case
        if (len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        # if vertical
        if 45 < orientation < 135:
            # sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    # def process_lines(self, lines, img):
    def process_lines(self, lines):
        '''Main function for lines from cv.HoughLinesP() output merging
        for OpenCV 3
        lines -- cv.HoughLinesP() output
        img -- binary image
        '''
        lines_x = []
        lines_y = []
        # for every line of cv2.HoughLinesP()
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation < 135:
                lines_y.append(line_i)
            else:
                lines_x.append(line_i)

        lines_y = sorted(lines_y, key=lambda line: line[1])
        lines_x = sorted(lines_x, key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_x, lines_y]:
            if len(i) > 0:
                groups = self.merge_lines_pipeline_2(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_lines_segments1(group))

                merged_lines_all.extend(merged_lines)

        return merged_lines_all


class MeteorDetector:
    def __init__(self):
        # In order to detect satellites (or planes), we need to compare
        # the previous image and the current image.
        #
        # If one line in the previous image has the same/similar angel
        # with one line in the current image, we may consider them as
        # trails by satellite or plane.
        #   -- Do we need to have a threshold how faraway these two lines
        #      can be considered as separated events?
        #      This is a question needs further monitoring
        #
        # So the detection logic would be:
        # 1) Detect lines in the first image (with subtraction from the 2nd
        #    image
        # 2) Store the detection lines, but not generate the extracted images
        #    at this point
        # 3) Detect lines for the 2nd image
        # 4) Compare the detection lines in the previous image and those
        #    from the current image
        # 5) If suspicious satellite detected:
        #    a) For the one belongs to the previous image, put it to the
        #       "previous_satellite"
        #    b) Put the satellite in the current image to the "current_satellite"
        # 6) After checking done:
        #    a) For the previous image, generate a list with "previous_satellite"
        #       removed. And then generate the extracted images
        #    b) For the list in the current image:
        #       i.  Current detection list -> self.previous_detection_lines
        #       ii. "current_satellite" -> self."previous_satellite"
        self.Previous_Image_Detection_Lines = []
        self.Previous_Image_Satellites = []
        self.Previous_Image_Filename = ''
        self.Previous_Image = []

        self.Current_Image_Detection_Lines = []
        self.Current_Image_Satellites = []

    # When two lines have the similar angel, get the two closest points
    # This is for next step to determine if these two lines are in the
    # same line
    def __get_most_close_two_points_from_two_lines_with_same_angel(self,
                                                                   L1_x1, L1_y1, L1_x2, L1_y2,
                                                                   L2_x1, L2_y1, L2_x2, L2_y2,
                                                                   angel):
        close_x1 = 0
        close_y1 = 0
        close_x2 = 0
        close_y2 = 0

        if angel >= 0:
            if min(L1_x1, L1_x2) < min(L2_x1, L2_x2):
                close_x1 = max(L1_x1, L1_x2)
                close_y1 = max(L1_y1, L1_y2)
                close_x2 = min(L2_x1, L2_x2)
                close_y2 = min(L2_y1, L2_y2)
            else:
                close_x1 = min(L1_x1, L1_x2)
                close_y1 = min(L1_y1, L1_y2)
                close_x2 = max(L2_x1, L2_x2)
                close_y2 = max(L2_y1, L2_y2)
        else:
            if min(L1_x1, L1_x2) < min(L2_x1, L2_x2):
                close_x1 = max(L1_x1, L1_x2)
                close_y1 = min(L1_y1, L1_y2)
                close_x2 = min(L2_x1, L2_x2)
                close_y2 = max(L2_y1, L2_y2)
            else:
                close_x1 = min(L1_x1, L1_x2)
                close_y1 = max(L1_y1, L1_y2)
                close_x2 = max(L2_x1, L2_x2)
                close_y2 = min(L2_y1, L2_y2)

        return close_x1, close_y1, close_x2, close_y2

    # Even though we pass he angle as parameter here, we expect the
    # angles of these two lines are already compared and are quite
    # closed, can be considered as equal
    def __calculate_two_parallel_lines_distance(self, L1_x_mid, L1_y_mid, L2_x_mid, L2_y_mid, angle):
        angle_mid = math.atan2((L2_y_mid - L1_y_mid), (L2_x_mid - L1_x_mid))

        # To ensure the angle range is (-pi/2, pi/2)
        if angle_mid > np.pi / 2:
            angle_mid = angle_mid - np.pi
        if angle_mid < -np.pi / 2:
            angle_mid = np.pi + angle_mid

        angle_mid = abs(angle_mid)

        angle_mid_to_line = abs(angle_mid - abs(angle))

        dist_mid = math.sqrt((L2_x_mid - L1_x_mid) ** 2 + (L2_y_mid - L1_y_mid) ** 2)
        vertical_dist = dist_mid * math.sin(angle_mid_to_line)

        return vertical_dist

    '''
    # Don't use this. 
    # A short line in a big photo image could cause the bias value 
    # of the line function (y=ax + b), varies too much    
    def __calculate_two_parallel_lines_distance(self, L1_x1, L1_y1, L1_x2, L1_y2, L2_x1, L2_y1, L2_x2, L2_y2, angle):
        if L1_x1 == L1_x2 or L2_x1 == L2_x2:
            # Vertical lines
            return abs(L2_x1 - L1_x1)

        if L1_y1 == L1_y2 or L1_y1 == L2_y2:
            # Horizontal lines
            return abs(L2_y1 - L1_y1)

        bias_1 = ((L1_x2 * L1_y1) - (L1_x1 * L1_y2)) / (L1_x2 - L1_x1)
        bias_2 = ((L2_x2 * L2_y1) - (L2_x1 * L2_y2)) / (L2_x2 - L2_x1)

        dist = abs((bias_1 - bias_2) * math.cos(angle))
        return dist
    '''

    # A. Some times one line could be recognized as two
    #    We want to get them merged to be one
    # B. Some times lines from two images could belong to the same satellite
    #    We want to distinguish them
    #
    # Criteria:
    #   1) Angle is almost the same
    #   2) Vertical distance is very close
    #   3) No overlap, like these:
    #         --------
    #              -------
    #      Only accept lines like these:
    #         -------- -----
    #   4) The distance between the two closest points
    #      is within threshold
    def __decide_if_two_lines_should_belong_to_the_same_object(self,
                                                               L1_x1, L1_y1, L1_x2, L1_y2,
                                                               L2_x1, L2_y1, L2_x2, L2_y2,
                                                               for_satellite=False):
        angle_L1 = math.atan2((L1_y2 - L1_y1), (L1_x2 - L1_x1))

        # To ensure the angle range is (-pi/2, pi/2)
        if angle_L1 > np.pi / 2:
            angle_L1 = angle_L1 - np.pi
        if angle_L1 < -np.pi / 2:
            angle_L1 = np.pi + angle_L1

        angle_L2 = math.atan2((L2_y2 - L2_y1), (L2_x2 - L2_x1))
        if angle_L2 > np.pi / 2:
            angle_L2 = angle_L2 - np.pi
        if angle_L2 < -np.pi / 2:
            angle_L2 = np.pi + angle_L2

        # Sometimes the two lines' direction is similar, but the angles
        # are reverted, like one is -80 deg and one is +80 deg.
        # In this case the delta should be calculated as 20 deg, not 160
        angle_delta = abs(angle_L1 - angle_L2)
        if angle_delta > np.pi/2:
            angle_delta = np.pi - angle_delta

        # if abs(angle_L1 - angle_L2) > settings.LINE_ANGEL_DELTA_THRESHOLD:
        if angle_delta > settings.LINE_ANGEL_DELTA_THRESHOLD:
            # Angle delta is too much
            return False

        L1_x_mid = int((L1_x1 + L1_x2) / 2)
        L1_y_mid = int((L1_y1 + L1_y2) / 2)
        L2_x_mid = int((L2_x1 + L2_x2) / 2)
        L2_y_mid = int((L2_y1 + L2_y2) / 2)

        angle_avg = (angle_L1 + angle_L2) / 2
        # vertical_dist = self.__calculate_two_parallel_lines_distance(L1_x1, L1_y1, L1_x2, L1_y2,
        #                                                              L2_x1, L2_y1, L2_x2, L2_y2,
        #                                                              angle_avg)
        vertical_dist = self.__calculate_two_parallel_lines_distance(L1_x_mid, L1_y_mid, L2_x_mid, L2_y_mid, angle_avg)

        if not for_satellite:
            # Check lines in the same image
            if vertical_dist > settings.LINE_VERTICAL_DISTANCE_FOR_MERGE_THRESHOLD:
                # Can only be considered as two parallel lines
                # Not to merge
                return False
        else:
            # Check line in different images for satellite detection
            if vertical_dist > settings.LINE_VERTICAL_DISTANCE_FOR_SATELLITE_THRESHOLD:
                # Can only be considered as two parallel lines
                # Not to merge
                return False

        # Checking for overlap
        b_overlap = False
        if min(L1_y1, L1_y2) < min(L2_y1, L2_y2):
            if max(L1_y2, L1_y2) > min(L2_y1, L2_y2):
                # return False
                b_overlap = True

        if min(L1_y1, L1_y2) > min(L2_y1, L2_y2):
            if min(L1_y1, L1_y2) < max(L2_y1, L2_y2):
                # return False
                b_overlap = True

        if min(L1_x1, L1_x2) < min(L2_x1, L2_x2):
            if max(L1_x1, L1_x2) > min(L2_x1, L2_x2):
                # return False
                b_overlap = True

        if min(L1_x1, L1_x2) > min(L2_x1, L2_x2):
            if min(L1_x1, L1_x2) < max(L2_x1, L2_x2):
                # return False
                b_overlap = True

        # If there's overlap, but are very close, merge them as well
        # In this case we don't need to calculate the closest two points
        # Just return true to merge them
        if b_overlap and vertical_dist <= settings.LINE_VERTICAL_DISTANCE_FOR_MERGE_W_OVERLAP_THRESHOLD:
            return True

        # Finally, check the most close two points
        close_x1, close_y1, close_x2, close_y2 = \
            self.__get_most_close_two_points_from_two_lines_with_same_angel(L1_x1, L1_y1, L1_x2, L1_y2,
                                                                            L2_x1, L2_y1, L2_x2, L2_y2,
                                                                            angle_avg)
        dist_close = math.sqrt((close_x2 - close_x1) ** 2 + (close_y2 - close_y1) ** 2)

        if not for_satellite:
            if dist_close > settings.LINE_DISTANCE_FOR_MERGE_THRESHOLD:
                return False
        else:
            if dist_close > settings.LINE_DISTANCE_FOR_SATELLITE_THRESHOLD:
                return False

        # All checking passed
        return True

    # The initial detection lines need some filtering/processing:
    # 1) There would be some false detection due to the original
    #    image border, with image rotated by star-alignment
    # 2) Some lines could be detected as two, or more. Try to
    #    merge them back to one
    def detection_lines_filtering(self, detection_lines, orig_image):
        filtered_false_detection = []

        height, width, channels = orig_image.shape

        # Step 1: Check if the lines are due to false detection
        #         from the border (original image border)

        for i, line in enumerate(detection_lines):

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]

            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # In some cases the image border could cause some
            # false detection. Remove these
            if abs(x1 - x2) < settings.LINE_X_OR_Y_DELTA_THRESHOLD:
                if min(x1, x2) <= settings.DETECTION_IMAGE_BORDER_THRESHOLD \
                        or max(x1, x2) >= width - settings.DETECTION_IMAGE_BORDER_THRESHOLD:
                    # ignore this line, should be border
                    continue

            if abs(y1 - y2) < settings.LINE_X_OR_Y_DELTA_THRESHOLD:
                if min(y1, y2) <= settings.DETECTION_IMAGE_BORDER_THRESHOLD \
                        or max(y1, y2) >= height - settings.DETECTION_IMAGE_BORDER_THRESHOLD:
                    # ignore this line, should be border
                    continue

            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # For images got rotated (for star-alignment propose), the original
            # edge of the image could cause some false detection
            #
            # This method is to get the color of some pixels around the center of
            # the detected line. If one pixel is all dark ([0,0,0]), then this
            # line should be the original edge
            x_mid = int((x1 + x2) / 2)
            y_mid = int((y1 + y2) / 2)

            # (x_mid, y_mid - 5), (x_mid, y_mid + 5)
            # (x_mid - 5, y_mid), (x_mid + 5, y_mid)

            # image[y, x, c]

            # line_center_radius_checking = 20

            color_up = orig_image[max(y_mid - settings.LINE_CENTER_RADIUS_CHECKING, 0), x_mid]
            color_down = orig_image[min(y_mid + settings.LINE_CENTER_RADIUS_CHECKING, height - 1), x_mid]
            color_left = orig_image[y_mid, max(x_mid - settings.LINE_CENTER_RADIUS_CHECKING, 0)]
            color_right = orig_image[y_mid, min(x_mid + settings.LINE_CENTER_RADIUS_CHECKING, width - 1)]

            # Initially we want to check if the color is [0,0,0]
            # However it proved that if the image just rotated a little bit, then
            # we cannot go very far away of the border, then sometimes the value
            # is not fully ZERO.
            # It could be like [2,0,0], [0,0,1], ...
            #
            # Let's set a threshold like [3,3,3]
            boo_up = color_up[0] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                     and color_up[1] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                     and color_up[2] < settings.DETECTION_BORDER_COLOR_THRESHOLD

            boo_down = color_down[0] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                     and color_down[1] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                     and color_down[2] < settings.DETECTION_BORDER_COLOR_THRESHOLD

            boo_left = color_left[0] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                       and color_left[1] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                       and color_left[2] < settings.DETECTION_BORDER_COLOR_THRESHOLD

            boo_right = color_right[0] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                       and color_right[1] < settings.DETECTION_BORDER_COLOR_THRESHOLD \
                       and color_right[2] < settings.DETECTION_BORDER_COLOR_THRESHOLD

            # if np.any(color_up) == 0 or np.any(color_down) == 0
            #   or np.any(color_left) == 0 or np.any(color_right) == 0:
            if boo_up or boo_down or boo_left or boo_right:
                # Should be at the original image edge
                # (due to image rotation by star-alignment)
                # Ignore this one

                # print("Original image edge. Ignored.")
                continue

            # Calculate the angel of the line, to help the merging in next step

            # The angel is calculated in rad
            angle = 0

            # if x1 == x2:
            #     # angle = 90
            #     angle = math.pi/2
            # else:
            #     angle = math.atan(abs(y2-y1)/abs(x2-x1))
            #
            # The atan2() can handle the case when (x2-x1)==0
            # The angel value can be positive or negative to
            # stand for different direction
            angle = math.atan2((y2 - y1), (x2 - x1))

            # To ensure the angle range is (-pi/2, pi/2)
            if angle > np.pi / 2:
                angle = angle - np.pi
            if angle < -np.pi / 2:
                angle = np.pi + angle

            filtered_false_detection.append([x1, y1, x2, y2, x_mid, y_mid, angle])
            # filtered_false_detection.append([x1, y1, x2, y2, angle])

        # Step 2:
        # For some real lines, they could be recognized as two short ones during detection
        # Try to merge these back to one
        # The method is:
        # 1) Calculate the angel of the line (done)
        # 2) If two lines have the similar angel, then get the two most close points
        #    a) If the distance of these two points are close enough, and
        #    b) If the angle of these two points is also similar to the angles
        #       of the original two lines
        #    then consider these two lines can be merged

        merged_detection = []

        # for i in range(len(filtered_false_detection)-1):
        for i in range(len(filtered_false_detection)):
            angle_1 = filtered_false_detection[i][6]

            if angle_1 == -3.14:
                # Such value was filled by below algorithm
                # Skip this line
                continue

            for j in range(i + 1, len(filtered_false_detection)):
                angle_2 = filtered_false_detection[j][6]

                if angle_2 == -3.14:
                    continue

                i_x1 = filtered_false_detection[i][0]
                i_y1 = filtered_false_detection[i][1]
                i_x2 = filtered_false_detection[i][2]
                i_y2 = filtered_false_detection[i][3]

                j_x1 = filtered_false_detection[j][0]
                j_y1 = filtered_false_detection[j][1]
                j_x2 = filtered_false_detection[j][2]
                j_y2 = filtered_false_detection[j][3]

                if self.__decide_if_two_lines_should_belong_to_the_same_object(i_x1, i_y1, i_x2, i_y2,
                                                                               j_x1, j_y1, j_x2, j_y2,
                                                                               for_satellite=False):
                    merged_x1 = 0
                    merged_y1 = 0
                    merged_x2 = 0
                    merged_y2 = 0

                    if angle_1 >= 0:
                        merged_x1 = min(i_x1, i_x2, j_x1, j_x2)
                        merged_y1 = min(i_y1, i_y2, j_y1, j_y2)
                        merged_x2 = max(i_x1, i_x2, j_x1, j_x2)
                        merged_y2 = max(i_y1, i_y2, j_y1, j_y2)
                    else:
                        merged_x1 = min(i_x1, i_x2, j_x1, j_x2)
                        merged_y1 = max(i_y1, i_y2, j_y1, j_y2)
                        merged_x2 = max(i_x1, i_x2, j_x1, j_x2)
                        merged_y2 = min(i_y1, i_y2, j_y1, j_y2)

                        # merged_detection.append([merged_x1, merged_y1, merged_x2, merged_y2, angle_1])

                        # The merged line to be updated to filtered_false_detection[i]
                        # And filtered_false_detection[j] is removed

                    new_angel = (angle_1 + angle_2) / 2

                    x_mid_merged = int((merged_x1 + merged_x2) / 2)
                    y_mid_merged = int((merged_y1 + merged_y2) / 2)

                    filtered_false_detection[i][0] = merged_x1
                    filtered_false_detection[i][1] = merged_y1
                    filtered_false_detection[i][2] = merged_x2
                    filtered_false_detection[i][3] = merged_y2
                    filtered_false_detection[i][4] = x_mid_merged
                    filtered_false_detection[i][5] = y_mid_merged
                    filtered_false_detection[i][6] = new_angel

                    filtered_false_detection[j][0] = 0
                    filtered_false_detection[j][1] = 0
                    filtered_false_detection[j][2] = 0
                    filtered_false_detection[j][3] = 0
                    filtered_false_detection[j][4] = 0
                    filtered_false_detection[j][5] = 0
                    # Use such an angle to indicate this is removed
                    filtered_false_detection[j][6] = -3.14

            # End of the j for loop
            # One entry in the i for loop has completely matched with others
            merged_detection.append([filtered_false_detection[i][0],
                                     filtered_false_detection[i][1],
                                     filtered_false_detection[i][2],
                                     filtered_false_detection[i][3],
                                     filtered_false_detection[i][4],
                                     filtered_false_detection[i][5],
                                     filtered_false_detection[i][6]
                                     ])
        return merged_detection

    # Get the two points coordinators for a square that can hold
    # the detected meteor image
    def get_box_coordinate_from_detected_line(self, x1, y1, x2, y2, img_width, img_height, factor=1):

        sample_width = abs(x2 - x1)
        sample_height = abs(y2 - y1)

        x_midpoint = int((x1 + x2) / 2)
        y_midpoint = int((y1 + y2) / 2)

        # if sample_width < square_size:

        # increase the area with a factor. Normally 1.5
        # sample_width = int(sample_width * settings.DETECTION_CROP_IMAGE_BOX_FACTOR)
        sample_width = int(sample_width * factor)
        # sample_height = int(sample_height * settings.DETECTION_CROP_IMAGE_BOX_FACTOR)
        sample_height = int(sample_height * factor)

        # The size can be at least 640 x 640
        # If the detected line size exceeds 640 pixels, the
        # draw box size can be enlarged accordingly
        draw_size = max(sample_width, sample_height)
        draw_size = max(draw_size, settings.DETECTION_CROP_IMAGE_BOX_SIZE)

        draw_x1 = x_midpoint - int(draw_size / 2)
        draw_x2 = x_midpoint + int(draw_size / 2)

        # Just make it be the exactly the same
        # size as the draw_size
        if draw_x2 - draw_x1 < draw_size:
            draw_x2 = draw_x1 + draw_size

        draw_y1 = y_midpoint - int(draw_size / 2)
        draw_y2 = y_midpoint + int(draw_size / 2)
        if draw_y2 - draw_y1 < draw_size:
            draw_y2 = draw_y1 + draw_size

        # Detect if exceed the img size, or smaller than 0
        # Here we didn't consider the exceptional case that
        # the draw box size exceeds the image size
        if draw_x1 < 0:
            draw_x2 = draw_x2 - draw_x1
            draw_x1 = 0

        if draw_x2 > img_width - 1:
            draw_x1 = draw_x1 - (draw_x2 - img_width + 1)
            draw_x2 = img_width - 1

        if draw_y1 < 0:
            draw_y2 = draw_y2 - draw_y1
            draw_y1 = 0

        if draw_y2 > img_height - 1:
            draw_y1 = draw_y1 - (draw_y2 - img_height + 1)
            draw_y2 = img_height - 1

        return draw_x1, draw_y1, draw_x2, draw_y2

    # NOTE:
    # ==================================================================
    # As we also need to merge some boxes which have overlap, this
    # function is to be deprecated !!!
    def get_box_list_from_meteor_lines(self, detection_lines, img_width, img_height):
        box_list = []

        for line in detection_lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]

            # cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            box_x1, box_y1, box_x2, box_y2 = \
                self.get_box_coordinate_from_meteor_line(x1, y1, x2, y2, img_width, img_height)

            box_list.append([box_x1, box_y1, box_x2, box_y2])

        return box_list

    def get_combined_box_list_from_detected_lines(self, detection_lines, img_width, img_height):
        # Step 1: Get the box list for each line
        # Step 2: If two boxes have overlap, combine them (enlarged)

        box_list = []

        for line in detection_lines:
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]

            # cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            box_x1, box_y1, box_x2, box_y2 = \
                self.get_box_coordinate_from_detected_line(x1, y1, x2, y2, img_width, img_height,
                                                           factor=settings.DETECTION_CROP_IMAGE_BOX_FACTOR)

            # The "True" value in the end is for next step usage
            # When two boxes are merged, tag one as False, indicating
            # it is to be removed
            box_list.append([box_x1, box_y1, box_x2, box_y2, True])

        combined_box_list = []

        for i in range(len(box_list)):

            tag = box_list[i][4]
            if not tag:
                # This has been merged, skip it
                continue

            # This algorithm still has bug, j should be started from 0
            # as well, just skipping the case of i==j
            # But if we get too many boxes merged together there could
            # be another problem. The image area could be quite big,
            # and could include both landscape objects and meteor objects.
            for j in range(i + 1, len(box_list)):
                tag = box_list[j][4]
                if not tag:
                    # This has been merged, skip it
                    continue

                # Because the value in the box_list[i] could be
                # updated, need to get the values from each loop
                i_x1 = box_list[i][0]
                i_y1 = box_list[i][1]
                i_x2 = box_list[i][2]
                i_y2 = box_list[i][3]

                i_x_mid = int((i_x1 + i_x2) / 2)
                i_y_mid = int((i_y1 + i_y2) / 2)

                i_width = abs(i_x2 - i_x1)

                j_x1 = box_list[j][0]
                j_y1 = box_list[j][1]
                j_x2 = box_list[j][2]
                j_y2 = box_list[j][3]

                j_x_mid = int((j_x1 + j_x2) / 2)
                j_y_mid = int((j_y1 + j_y2) / 2)

                j_width = abs(j_x2 - j_x1)

                # Compare the center distance in each dimension
                # If the distances in both dimensions are all
                # less than the two boxes' width/2 (sum up), then
                # it is a overlap
                center_dist_x = abs(j_x_mid - i_x_mid)
                center_dist_y = abs(j_y_mid - i_y_mid)

                if center_dist_x < (i_width + j_width)/2 * settings.BOX_OVERLAP_THRESHOLD\
                        and center_dist_y < (i_width + j_width)/2 * settings.BOX_OVERLAP_THRESHOLD:
                    # Overlap detected
                    merged_x1 = min([i_x1, i_x2, j_x1, j_x2])
                    merged_y1 = min([i_y1, i_y2, j_y1, j_y2])
                    merged_x2 = max([i_x1, i_x2, j_x1, j_x2])
                    merged_y2 = max([i_y1, i_y2, j_y1, j_y2])

                    # Make it as a square, not a rectangle
                    # Parameter (factor = 1) means no extra extension
                    #
                    # merged_width = max(abs(merged_x2 - merged_x1), abs(merged_y2 - merged_y1))
                    merged_x1, merged_y1, merged_x2, merged_y2 = \
                        self.get_box_coordinate_from_detected_line(merged_x1,
                                                                   merged_y1,
                                                                   merged_x2,
                                                                   merged_y2,
                                                                   img_width,
                                                                   img_height,
                                                                   factor=1)

                    box_list[i][0] = merged_x1
                    box_list[i][1] = merged_y1
                    box_list[i][2] = merged_x2
                    box_list[i][3] = merged_y2

                    box_list[j][0] = 0
                    box_list[j][1] = 0
                    box_list[j][2] = 0
                    box_list[j][3] = 0
                    box_list[j][4] = False

            # End of the j loop
            # One entry in the i for loop has completely merged with others
            combined_box_list.append([box_list[i][0], box_list[i][1], box_list[i][2], box_list[i][3]])

        return combined_box_list

    def detect_meteor_from_image(self, detection_img, original_img):

        # To ensure we have an odd value for he kernel size
        blur_kernel_size = settings.DETECTION_BLUR_KERNEL_SIZE
        count = blur_kernel_size % 2
        if count == 0:
            blur_kernel_size += 1

        blur_img = cv2.GaussianBlur(detection_img, (blur_kernel_size, blur_kernel_size), 0)

        # blur_img_enh = cv2.GaussianBlur(enhanced_img, (blur_kernel_size, blur_kernel_size), 0)

        canny_lowThreshold = settings.DETECTION_CANNY_LOW_THRESHOLD
        canny_ratio = settings.DETECTION_CANNY_RATIO
        canny_kernel_size = settings.DETECTION_CANNY_KERNEL_SIZE

        detected_edges = cv2.Canny(blur_img,
                                   canny_lowThreshold,
                                   canny_lowThreshold * canny_ratio,
                                   apertureSize=canny_kernel_size)

        line_threshold = settings.DETECTION_LINE_THRESHOLD
        minLineLength = settings.DETECTION_LINE_MIN_LINE_LENGTH
        maxLineGap = settings.DETECTION_LINE_MAX_LINE_GAP

        lines = cv2.HoughLinesP(image=detected_edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=line_threshold,
                                minLineLength=minLineLength,
                                maxLineGap=maxLineGap)

        if not (lines is None):
            my_HoughBundler = HoughBundler()

            # Merge those lines that are very closed
            # Should consider them as just one
            #
            # The format for each "line" element in the list would be like this:
            # [array([1565, 3099], dtype=int32), array([1663, 2986], dtype=int32)]
            #
            # And can be accessed in this way:
            #
            # line[0] = [1565 3099]
            # line[1] = [1663 2986]
            #
            # x1 = line[0][0]
            # y1 = line[0][1]
            # x2 = line[1][0]
            # y2 = line[1][1]

            merged_lines = my_HoughBundler.process_lines(lines=lines)

            # Remove some false detection.
            # And merging some lines which could belong to be same one.
            # The original image is needed (for some color detection)
            filtered_lines = self.detection_lines_filtering(merged_lines, original_img)
            return filtered_lines
        else:
            return None

    def draw_detection_boxes_on_image(self, original_img, detection_lines, color):
        # Get the detected lines coordinates
        # detection_lines = self.detect_meteor_from_image(original_img)

        draw_img = copy.copy(original_img)
        height, width, channels = draw_img.shape

        # box_list = self.get_box_list_from_meteor_lines(detection_lines, width, height)
        box_list = self.get_combined_box_list_from_detected_lines(detection_lines, width, height)
        # print(box_list)

        for line in detection_lines:
            # the format for "line" here would be like this:
            # [array([1565, 3099], dtype=int32), array([1663, 2986], dtype=int32)]
            # line[0] = [1565 3099]
            # line[1] = [1663 2986]

            # x1 = line[0][0]
            # y1 = line[0][1]
            # x2 = line[1][0]
            # y2 = line[1][1]

            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]

            angle = line[6]

            cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.putText(draw_img, '{0:.3f}'.format(angle * 180 / np.pi),
                        (x2 + 10, y2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=(255, 255, 255),
                        lineType=2)

            # draw_x1, draw_y1, draw_x2, draw_y2 = \
            #     self.get_box_coordinate_from_meteor_line(x1, y1, x2, y2, width, height)
            # cv2.rectangle(draw_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (255, 255, 0), 1)

        for box in box_list:
            # for draw_x1, draw_y1, draw_x2, draw_y2 in box:
            box_x1 = box[0]
            box_y1 = box[1]
            box_x2 = box[2]
            box_y2 = box[3]
            # cv2.rectangle(draw_img, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 0), 1)
            cv2.rectangle(draw_img, (box_x1, box_y1), (box_x2, box_y2), color, 1)

        return draw_img

    # The "orig_filename" parameter should not have path info,
    # just the file name like "xxxx.jpg"
    #
    # The extracted image will be saved with file name like this:
    #     ER4A3109_0001_pos_(02194,02421)_(02834,03061).JPG
    #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
    # The size(x,y) is the original size of the photo
    # The pos (x1, y1) (x2, y2) position from original image is kept in the file name
    #
    def extract_meteor_images_to_file(self, original_img, detection_lines, save_dir, orig_filename, verbose):
        # When calling this function, need to do such validation:
        # if not (detection_lines is None):
        #   ...

        height, width, channels = original_img.shape

        # box_list = self.get_box_list_from_meteor_lines(detection_lines, width, height)
        box_list = self.get_combined_box_list_from_detected_lines(detection_lines, width, height)
        i = 0

        # filename_no_ext = os.path.splitext(orig_filename)[0]
        filename_no_ext, file_ext = os.path.splitext(orig_filename)

        for box in box_list:
            # for draw_x1, draw_y1, draw_x2, draw_y2 in box:
            box_x1 = box[0]
            box_y1 = box[1]
            box_x2 = box[2]
            box_y2 = box[3]

            crop_img = original_img[box_y1:box_y2, box_x1:box_x2]

            i += 1

            # Let the file name to contain the position info
            # So as to know where it is from the original image
            file_to_save = filename_no_ext +\
                           "_size_({:05d},{:05d})_{:04d}_pos_({:05d},{:05d})_({:05d},{:05d})".\
                               format(width, height, i, box_x1, box_y1, box_x2, box_y2) + \
                           file_ext

            file_to_save = os.path.join(save_dir, file_to_save)
            if verbose:
                print("    saving {} ...".format(file_to_save))

            cv2.imwrite(file_to_save, crop_img)
        # End of function

    # Logic:
    # 1) For each element in the self.Previous_Image_Detection_List,
    #    compare with all items in the self.Current_Image_Detection_List
    # 2) If the two have similar angles, and
    #    if the line between them, connecting the most closed two points
    #    have the similar angle, then consider these two are from the
    #    same satellite object
    def check_satellite_with_previous_detection_list(self, verbose):
        for previous_line in self.Previous_Image_Detection_Lines:
            p_x1 = previous_line[0]
            p_y1 = previous_line[1]
            p_x2 = previous_line[2]
            p_y2 = previous_line[3]
            p_x_mid = previous_line[4]
            p_y_mid = previous_line[5]
            p_angle = previous_line[6]

            for current_line in self.Current_Image_Detection_Lines:
                c_x1 = current_line[0]
                c_y1 = current_line[1]
                c_x2 = current_line[2]
                c_y2 = current_line[3]
                c_x_mid = current_line[4]
                c_y_mid = current_line[5]
                c_angle = current_line[6]

                if self.__decide_if_two_lines_should_belong_to_the_same_object(p_x1, p_y1, p_x2, p_y2,
                                                                               c_x1, c_y1, c_x2, c_y2,
                                                                               for_satellite=True):
                    # Ok we can consider these as due to the same satellite
                    self.Previous_Image_Satellites.append([p_x1, p_y1, p_x2, p_y2, p_x_mid, p_y_mid, p_angle])
                    self.Current_Image_Satellites.append([c_x1, c_y1, c_x2, c_y2, c_x_mid, c_y_mid, c_angle])

            # end of the for current_line loop
        # end of the for previous_line loop

        if verbose:
            print('... {} detected {} satellites'.format(self.Previous_Image_Filename,
                                                         len(self.Previous_Image_Satellites)))
        # End of function

    # Logic:
    # 1) Detect the lines from current image (orig_filename)
    # 2) Compare the current detection list with the stored
    #    previous detection list, to detect if any possible
    #    satellites
    # 3) Exclude the possible satellites from the previous
    #    detection list
    # 4) Extract the detection objects from the previous image
    def detect_n_process_the_previous_image(self, file_dir, orig_filename, save_dir, file_for_subtraction, verbose):
        # Directory to save the image drawn with detection boxes
        draw_box_file_dir = os.path.join(save_dir, '1_detection')
        if not os.path.exists(draw_box_file_dir):
            os.mkdir(draw_box_file_dir)

        # Directory to save the extracted small images
        extracted_file_dir = os.path.join(save_dir, '2_cropped')
        if not os.path.exists(extracted_file_dir):
            os.mkdir(extracted_file_dir)

        filename_w_path = os.path.join(file_dir, orig_filename)
        orig_img = cv2.imread(filename_w_path)

        # Do a subtraction with another image, which has been star-aligned
        file_for_subtraction_w_path = os.path.join(file_dir, file_for_subtraction)
        img_for_subtraction = cv2.imread(file_for_subtraction_w_path)
        img = cv2.subtract(orig_img, img_for_subtraction)

        detection_lines = self.detect_meteor_from_image(img, orig_img)
        if not (detection_lines is None):
            self.Current_Image_Detection_Lines = detection_lines
        else:
            self.Current_Image_Detection_Lines = []

        # Check if we are the first image.
        # Only when we are not the first image we'll do the extraction
        if len(self.Previous_Image_Filename) > 0:
            # We can now extract the detection objects from the
            # previous image, with excluding the possible satellites

            # The self.Previous_Image_Detection_Lines,
            #     self.Previous_Image_Satellites,
            #     self.Current_Image_Satellites
            # will be updated
            self.check_satellite_with_previous_detection_list(verbose)

            filename_no_ext, file_ext = os.path.splitext(self.Previous_Image_Filename)

            previous_detection_wo_satellite = []
            for line in self.Previous_Image_Detection_Lines:
                if not (line in self.Previous_Image_Satellites):
                    previous_detection_wo_satellite.append([line[0], line[1], line[2], line[3],
                                                            line[4], line[5], line[6]])

            print(self.Previous_Image_Filename)
            print(self.Previous_Image_Detection_Lines)
            print(self.Previous_Image_Satellites)
            print(previous_detection_wo_satellite)

            # if len(previous_detection_wo_satellite) > 0:
            if len(self.Previous_Image_Detection_Lines) > 0:
                draw_img = self.draw_detection_boxes_on_image(self.Previous_Image,
                                                              previous_detection_wo_satellite,
                                                              color=(255, 255, 0))

                # Also highlight the possible satellites as well
                draw_img = self.draw_detection_boxes_on_image(draw_img,
                                                              self.Previous_Image_Satellites,
                                                              color=(0, 255, 255))

                draw_filename = filename_no_ext + "_detection_{}".format(len(previous_detection_wo_satellite))
                draw_filename = draw_filename + file_ext

                # draw_filename = os.path.join(file_dir, draw_filename)
                # draw_filename = os.path.join(save_dir, draw_filename)
                draw_filename = os.path.join(draw_box_file_dir, draw_filename)
                cv2.imwrite(draw_filename, draw_img)

                # Extract the detected portions to small image files
                # Normally they would be 640x640 size, but can be bigger
                self.extract_meteor_images_to_file(self.Previous_Image,
                                                   previous_detection_wo_satellite,
                                                   extracted_file_dir,
                                                   self.Previous_Image_Filename,
                                                   verbose)
            else:
                # No any line detected
                # Save an image with updated file name to indicate
                # detection is 0
                draw_filename = filename_no_ext + "_detection_0" + file_ext

                # draw_filename = os.path.join(save_dir, draw_filename)
                draw_filename = os.path.join(draw_box_file_dir, draw_filename)
                cv2.imwrite(draw_filename, self.Previous_Image)

        # The previous file was handled and done
        # Update the previous data to the current
        self.Previous_Image_Detection_Lines = self.Current_Image_Detection_Lines
        self.Previous_Image_Satellites = self.Current_Image_Satellites
        self.Previous_Image = copy.copy(orig_img)
        self.Previous_Image_Filename = orig_filename

        self.Current_Image_Detection_Lines = []
        self.Current_Image_Satellites = []
    # End of function

    # The "orig_filename" parameter should not have path info,
    # just the file name like "xxxx.jpg"
    #
    # The file_dir is the original image file folder
    # In the save_dir, two sub_folders should have been created:
    #
    # <save_dir>
    # .. 1_detection
    # .. 2_extraction
    #
    # If the "file_for_subtraction" parameter is not null, that means
    # we want to do a subtraction with that file, before doing detection.
    # This file should be put in the same folder.
    # -- This would help if the two images have been star-aligned
    # -- Doing a subtraction would make the moving objects be very clear
    #
    # NOTE:
    # ==================================================================
    # As we also need to detect satellites with two images, this function
    # is to be deprecated !!!
    def detect_n_extract_meteor_image_file(self, file_dir, orig_filename, save_dir, verbose, file_for_subtraction=''):
        # Directory to save the image drawn with detection boxes
        draw_box_file_dir = os.path.join(save_dir, '1_detection')
        if not os.path.exists(draw_box_file_dir):
            os.mkdir(draw_box_file_dir)

        # Directory to save the extracted small images
        extracted_file_dir = os.path.join(save_dir, '2_cropped')
        if not os.path.exists(extracted_file_dir):
            os.mkdir(extracted_file_dir)

        filename_w_path = os.path.join(file_dir, orig_filename)
        orig_img = cv2.imread(filename_w_path)

        filename_no_ext, file_ext = os.path.splitext(orig_filename)

        # Do a subtraction with another image, which has been star-aligned
        # with this one already
        if len(file_for_subtraction) > 0:
            file_for_subtraction_w_path = os.path.join(file_dir, file_for_subtraction)
            img_for_subtraction = cv2.imread(file_for_subtraction_w_path)
            img = cv2.subtract(orig_img, img_for_subtraction)

        # The subtracted image is purely used for line detection
        # The original image is still used for further filtering
        detection_lines = self.detect_meteor_from_image(img, orig_img, color=(255, 255, 0))
        if not (detection_lines is None):
            # print(detection_lines)

            '''
            # Each image files would have the extracted files to
            # be put to a sub-folder, which named with the image
            # file name
            extract_dir = os.path.join(save_dir, orig_filename)
            if not os.path.exists(extract_dir):
                os.mkdir(extract_dir)
            '''

            draw_img = self.draw_detection_boxes_on_image(orig_img, detection_lines)

            draw_filename = filename_no_ext + "_detection_{}".format(len(detection_lines)) + file_ext

            # draw_filename = os.path.join(file_dir, draw_filename)
            # draw_filename = os.path.join(save_dir, draw_filename)
            draw_filename = os.path.join(draw_box_file_dir, draw_filename)
            cv2.imwrite(draw_filename, draw_img)

            # Extract the detected portions to small image files
            # Normally they would be 640x640 size, but can be bigger
            self.extract_meteor_images_to_file(orig_img, detection_lines, extracted_file_dir, orig_filename, verbose)
        else:
            # No line detected
            # Save an image with updated file name to indicate
            # detection is 0
            draw_filename = filename_no_ext + "_detection_0" + file_ext

            # draw_filename = os.path.join(save_dir, draw_filename)
            draw_filename = os.path.join(draw_box_file_dir, draw_filename)
            cv2.imwrite(draw_filename, orig_img)

    # Go through all image files in the "file_dir". Will only
    # support ".jpg", ".tif" at this time.
    #
    def detect_n_extract_meteor_from_folder(self, file_dir, save_dir, subtraction=True, verbose=1):
        # image_list = fnmatch.filter(os.listdir(file_dir), '*.py')

        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']
        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Directory to save the image drawn with detection boxes
        draw_box_file_dir = os.path.join(save_dir, '1_detection')
        if not os.path.exists(draw_box_file_dir):
            os.mkdir(draw_box_file_dir)

        # Directory to save the extracted small images
        extracted_file_dir = os.path.join(save_dir, '2_cropped')
        if not os.path.exists(extracted_file_dir):
            os.mkdir(extracted_file_dir)

        num_of_images = len(image_list)
        for index, image_file in enumerate(image_list):
            if verbose:
                print("\nProcessing image {} ...".format(image_file))

            if subtraction and num_of_images > 1:
                # For star-aligned images, doing a subtraction will easily remove
                # most of the star background. Easy for detection
                # Recommended to use this method

                # Get the next image in the list for subtraction
                # If this image is the last one in the list, get the previous image then

                if index < num_of_images - 1:
                    next_image_file = image_list[index + 1]
                else:
                    next_image_file = image_list[index - 1]

                # NOTE:
                # This function is to be deprecated
                # self.detect_n_extract_meteor_image_file(file_dir, image_file, save_dir, verbose,
                #                                         file_for_subtraction=next_image_file)

                self.detect_n_process_the_previous_image(file_dir, image_file, save_dir,
                                                         file_for_subtraction=next_image_file,
                                                         verbose=verbose)
            else:
                # Detection without image subtraction
                # This would be rarely used now...
                self.detect_n_extract_meteor_image_file(file_dir, image_file, save_dir, verbose)

'''
if __name__ == "__main__":

    meteor_detector = MeteorDetector()

    filename = "IMG_3214_r.jpg"
    next_file_name = "IMG_3224_r.jpg"
    file_dir = 'F:/meteor-one-click-test-original-images/star-aligned'
    save_dir = 'F:/meteor-one-click-test-original-images/star-aligned/process'

    meteor_detector.detect_n_extract_meteor_image_file(file_dir, filename, save_dir, verbose=1,
                                                       file_for_subtraction=next_file_name)
    # meteor_detector.detect_n_extract_meteor_from_folder(file_dir, save_dir, verbose=1)

    # meteor_detector.get_box_coordinate_from_meteor_line(1954, 3618, 2027, 3611, 5437, 3625)

'''