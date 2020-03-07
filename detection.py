import cv2
import numpy as np
import copy
import math
import os, fnmatch

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

            '''
            cv2.putText(img, '{}'.format(i),
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        fontColor = (255,255,255),
                        lineType=1)

            '''
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

            # filtered_false_detection.append([x1, y1, x2, y2, x_mid, y_mid, angle])
            filtered_false_detection.append([x1, y1, x2, y2, angle])

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
            angle_1 = filtered_false_detection[i][4]

            if angle_1 == -3.14:
                # Such value was filled by below algorithm
                # Skip this line
                continue

            for j in range(i + 1, len(filtered_false_detection)):
                angle_2 = filtered_false_detection[j][4]

                if abs(angle_1 - angle_2) <= settings.LINE_ANGEL_DELTA_THRESHOLD:
                    # Get the most closed two points
                    i_x1 = filtered_false_detection[i][0]
                    i_y1 = filtered_false_detection[i][1]
                    i_x2 = filtered_false_detection[i][2]
                    i_y2 = filtered_false_detection[i][3]

                    j_x1 = filtered_false_detection[j][0]
                    j_y1 = filtered_false_detection[j][1]
                    j_x2 = filtered_false_detection[j][2]
                    j_y2 = filtered_false_detection[j][3]

                    close_x1 = 0
                    close_y1 = 0
                    close_x2 = 0
                    close_y2 = 0

                    if angle_1 >= 0:
                        if min(i_x1, i_x2) < min(j_x1, j_x2):
                            close_x1 = max(i_x1, i_x2)
                            close_y1 = max(i_y1, i_y2)
                            close_x2 = min(j_x1, j_x2)
                            close_y2 = min(j_y1, j_y2)
                        else:
                            close_x1 = min(i_x1, i_x2)
                            close_y1 = min(i_y1, i_y2)
                            close_x2 = max(j_x1, j_x2)
                            close_y2 = max(j_y1, j_y2)
                    else:
                        if min(i_x1, i_x2) < min(j_x1, j_x2):
                            close_x1 = max(i_x1, i_x2)
                            close_y1 = min(i_y1, i_y2)
                            close_x2 = min(j_x1, j_x2)
                            close_y2 = max(j_y1, j_y2)
                        else:
                            close_x1 = min(i_x1, i_x2)
                            close_y1 = max(i_y1, i_y2)
                            close_x2 = max(j_x1, j_x2)
                            close_y2 = min(j_y1, j_y2)

                    angle_close = math.atan2((close_y2 - close_y1), (close_x2 - close_x1))
                    dist_close = math.sqrt((close_x2 - close_x1) ** 2 + (close_y2 - close_y1) ** 2)

                    # dist_1 = math.sqrt((i_x1 - j_x1) ** 2 + (i_y1 - j_y1) ** 2)
                    # dist_2 = math.sqrt((i_x1 - j_x2) ** 2 + (i_y1 - j_y2) ** 2)
                    # dist_3 = math.sqrt((i_x2 - j_x1) ** 2 + (i_y2 - j_y1) ** 2)
                    # dist_4 = math.sqrt((i_x2 - j_x2) ** 2 + (i_y2 - j_y2) ** 2)

                    # dist_min = min([dist_1, dist_2, dist_3, dist_4])

                    if abs(angle_close - ((angle_1 + angle_2) / 2)) <= settings.LINE_ANGEL_DELTA_THRESHOLD:
                        if dist_close <= settings.LINE_DISTANCE_FOR_MERGE_THRESHOLD:
                            # These two lines are
                            # 1) closed enough
                            # 2) have the same angle
                            # 3) the connection lines also have the same angle
                            #
                            # Can be considered as one line

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

                            filtered_false_detection[i][0] = merged_x1
                            filtered_false_detection[i][1] = merged_y1
                            filtered_false_detection[i][2] = merged_x2
                            filtered_false_detection[i][3] = merged_y2
                            filtered_false_detection[i][4] = new_angel

                            filtered_false_detection[j][0] = 0
                            filtered_false_detection[j][1] = 0
                            filtered_false_detection[j][2] = 0
                            filtered_false_detection[j][3] = 0
                            # Use such an angle to indicate this is removed
                            filtered_false_detection[j][4] = -3.14

            # End of the j for loop
            # One entry in the i for loop has completely matched with others
            merged_detection.append([filtered_false_detection[i][0],
                                     filtered_false_detection[i][1],
                                     filtered_false_detection[i][2],
                                     filtered_false_detection[i][3],
                                     filtered_false_detection[i][4]
                                     ])
        return merged_detection

    # Get the two points coordinators for a square that can hold
    # the detected meteor image
    def get_box_coordinate_from_meteor_line(self, x1, y1, x2, y2, img_width, img_height, factor=1):

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

    def get_combined_box_list_from_meteor_lines(self, detection_lines, img_width, img_height):
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
                self.get_box_coordinate_from_meteor_line(x1, y1, x2, y2, img_width, img_height,
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

                if center_dist_x < (i_width + j_width)/2 and center_dist_y < (i_width + j_width)/2:
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
                        self.get_box_coordinate_from_meteor_line(merged_x1,
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
            # Those lines at the image edge, and in parallel with
            # the edge, are to be considered as false detection

            '''
            real_lines = []
            height, width, channels = original_img.shape
            for line in merged_lines:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[1][0]
                y2 = line[1][1]

                if abs(x1 == x2) < settings.DETECTION_LINE_X_OR_Y_DELTA_THRESHOLD \
                        and (max(x1, x2) <= settings.DETECTION_IMAGE_BORDER_THRESHOLD
                             or min(x1, x2) >= width - settings.DETECTION_IMAGE_BORDER_THRESHOLD):
                    # ignore this line, should be border
                    continue

                if abs(y1 == y2) < settings.DETECTION_LINE_X_OR_Y_DELTA_THRESHOLD \
                        and (max(y1, y2) <= settings.DETECTION_IMAGE_BORDER_THRESHOLD
                             or min(y1, y2) >= height - settings.DETECTION_IMAGE_BORDER_THRESHOLD):
                    # ignore this line, should be border
                    continue

                real_lines.append(line)
            
            return real_lines
            '''

            # Use the original image for further filtering
            filtered_lines = self.detection_lines_filtering(merged_lines, original_img)
            return filtered_lines
        else:
            return None

    def draw_detection_boxes_on_image(self, original_img, detection_lines):
        # Get the detected lines coordinates
        # detection_lines = self.detect_meteor_from_image(original_img)

        draw_img = copy.copy(original_img)
        height, width, channels = draw_img.shape

        # box_list = self.get_box_list_from_meteor_lines(detection_lines, width, height)
        box_list = self.get_combined_box_list_from_meteor_lines(detection_lines, width, height)
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

            cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # draw_x1, draw_y1, draw_x2, draw_y2 = \
            #     self.get_box_coordinate_from_meteor_line(x1, y1, x2, y2, width, height)
            # cv2.rectangle(draw_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (255, 255, 0), 1)

        for box in box_list:
            # for draw_x1, draw_y1, draw_x2, draw_y2 in box:
            box_x1 = box[0]
            box_y1 = box[1]
            box_x2 = box[2]
            box_y2 = box[3]
            cv2.rectangle(draw_img, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 0), 1)

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
        box_list = self.get_combined_box_list_from_meteor_lines(detection_lines, width, height)
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

    # The "orig_filename" parameter should not have path info,
    # just the file name like "xxxx.jpg"
    #
    '''
    # For each image file, a sub-folder will be created in the
    # "save_dir". The sub-folder will be named with the image
    # file name, like "xxxxx.jpg". Folder structure would be like
    # this:
    #
    # <save_dir>
    #       -- <xxxxx.jpg>
    #                -- extracted image files
    #
    '''
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
    def detect_n_extract_meteor_image_file(self, file_dir, orig_filename, save_dir, verbose, file_for_subtraction=''):
        # Directory to save the image drawn with detection boxes
        draw_box_file_dir = os.path.join(save_dir, '1_detection')
        if not os.path.exists(draw_box_file_dir):
            os.mkdir(draw_box_file_dir)

        # Directory to save the extracted small images
        extrated_file_dir = os.path.join(save_dir, '2_extraction')
        if not os.path.exists(extrated_file_dir):
            os.mkdir(extrated_file_dir)

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
        detection_lines = self.detect_meteor_from_image(img, orig_img)
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
            self.extract_meteor_images_to_file(orig_img, detection_lines, extrated_file_dir, orig_filename, verbose)
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
        extracted_file_dir = os.path.join(save_dir, '2_extraction')
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

                self.detect_n_extract_meteor_image_file(file_dir, image_file, save_dir, verbose,
                                                        file_for_subtraction=next_image_file)
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