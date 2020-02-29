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

    # Get the two points coordinators for a square that can hold
    # the detected meteor image
    def get_box_coordinate_from_meteor_line(self, x1, y1, x2, y2, img_width, img_height):

        sample_width = abs(x2 - x1)
        sample_height = abs(y2 - y1)

        x_midpoint = int((x1 + x2) / 2)
        y_midpoint = int((y1 + y2) / 2)

        # if sample_width < square_size:

        # increase the area with a factor. Normally 1.5
        sample_width = int(sample_width * settings.detection_crop_img_box_factor)
        sample_height = int(sample_height * settings.detection_crop_img_box_factor)

        # The size can be at least 640 x 640
        # If the detected line size exceeds 640 pixels, the
        # draw box size can be enlarged accordingly
        draw_size = max(sample_width, sample_height)
        draw_size = max(draw_size, settings.detection_crop_img_box_size)

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

    def detect_meteor_from_image(self, original_img):

        # To ensure we have an odd value for he kernel size
        blur_kernel_size = settings.detection_blur_kernel_size
        count = blur_kernel_size % 2
        if count == 0:
            blur_kernel_size += 1

        blur_img = cv2.GaussianBlur(original_img, (blur_kernel_size, blur_kernel_size), 0)

        # blur_img_enh = cv2.GaussianBlur(enhanced_img, (blur_kernel_size, blur_kernel_size), 0)

        canny_lowThreshold = settings.detection_canny_lowThreshold
        canny_ratio = settings.detection_canny_ratio
        canny_kernel_size = settings.detection_canny_kernel_size

        detected_edges = cv2.Canny(blur_img,
                                   canny_lowThreshold,
                                   canny_lowThreshold * canny_ratio,
                                   apertureSize=canny_kernel_size)

        line_threshold = settings.detection_line_threshold
        minLineLength = settings.detection_line_minLineLength
        maxLineGap = settings.detection_line_maxLineGap

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

            return merged_lines
        else:
            return None

    def draw_detection_boxes_on_image(self, original_img, detection_lines):
        # Get the detected lines coordinates
        # detection_lines = self.detect_meteor_from_image(original_img)

        draw_img = copy.copy(original_img)
        height, width, channels = draw_img.shape

        box_list = self.get_box_list_from_meteor_lines(detection_lines, width, height)
        # print(box_list)

        for line in detection_lines:
            # the format for "line" here would be like this:
            # [array([1565, 3099], dtype=int32), array([1663, 2986], dtype=int32)]
            # line[0] = [1565 3099]
            # line[1] = [1663 2986]

            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]

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

        box_list = self.get_box_list_from_meteor_lines(detection_lines, width, height)

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
    def detect_n_extract_meteor_image_file(self, file_dir, orig_filename, save_dir, verbose):
        # Directory to save the image drawn with detection boxes
        draw_box_file_dir = os.path.join(save_dir, '1_detection')

        # Directory to save the extracted small images
        extrated_file_dir = os.path.join(save_dir, '2_extraction')

        filename_w_path = os.path.join(file_dir, orig_filename)
        img = cv2.imread(filename_w_path)

        filename_no_ext, file_ext = os.path.splitext(orig_filename)

        detection_lines = self.detect_meteor_from_image(img)
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

            draw_img = self.draw_detection_boxes_on_image(img, detection_lines)

            draw_filename = filename_no_ext + "_detection_{}".format(len(detection_lines)) + file_ext

            # draw_filename = os.path.join(file_dir, draw_filename)
            # draw_filename = os.path.join(save_dir, draw_filename)
            draw_filename = os.path.join(draw_box_file_dir, draw_filename)
            cv2.imwrite(draw_filename, draw_img)

            # Extract the detected portions to small image files
            # Normally they would be 640x640 size, but can also bigger
            self.extract_meteor_images_to_file(img, detection_lines, extrated_file_dir, orig_filename, verbose)
        else:
            # No line detected
            # Save an image with updated file name to indicate
            # detection is 0
            draw_filename = filename_no_ext + "_detection_0" + file_ext

            # draw_filename = os.path.join(save_dir, draw_filename)
            draw_filename = os.path.join(draw_box_file_dir, draw_filename)
            cv2.imwrite(draw_filename, img)

    # Go through all image files in the "file_dir". Will only
    # support ".jpg", ".tif" at this time.
    #
    def detect_n_extract_meteor_from_folder(self, file_dir, save_dir, verbose):
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
        extrated_file_dir = os.path.join(save_dir, '2_extraction')
        if not os.path.exists(extrated_file_dir):
            os.mkdir(extrated_file_dir)

        for image_file in image_list:
            if verbose:
                print("\nProcessing image {} ...".format(image_file))

            # self.detect_n_extract_meteor_image_file(file_dir, image_file, extract_dir, verbos)
            self.detect_n_extract_meteor_image_file(file_dir, image_file, save_dir, verbose)


'''
if __name__ == "__main__":

    meteor_detector = MeteorDetector()

    filename = "meteor-shower.jpg"
    file_dir = 'data/meteor/meteor-data/meteor-detection/original_images/'
    save_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted'

    # meteor_detector.detect_n_extract_meteor_image_file(file_dir, filename, save_dir)
    meteor_detector.detect_n_extract_meteor_from_folder(file_dir, save_dir, verbose=1)

    # meteor_detector.get_box_coordinate_from_meteor_line(1954, 3618, 2027, 3611, 5437, 3625)

'''