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


# Deprecated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# This extends the XXX x XXX cropped mask image to the original big
# photo file size.
#
# The file name is like this:
#     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
#
# NOTE:
# ==================================================================
# As processing the extraction with original photo size is quite
# slow. This is to be deprecated !!!
def extend_mask_to_original_photo_size(self, file_dir, save_dir):
    print("\nExtending the mask back to original photo size ...")
    included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    image_list = [fn for fn in os.listdir(file_dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    for image_file in image_list:
        filename_w_path = os.path.join(file_dir, image_file)
        filename_no_ext, file_ext = os.path.splitext(image_file)

        # The file name would be like this:
        #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
        # The size (x, y) is the original photo size
        # The pos (x1, y1) (x2, y2) are the position from original image
        # Need to get these info back
        x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)
        target_width, target_height = self.get_image_size_from_file_name(image_file)

        img = cv2.imread(filename_w_path)

        left_extend = x1
        top_extend = y1
        bottom_extend = target_height - y2
        right_extend = target_width - x2

        extend_img = cv2.copyMakeBorder(img, top_extend, bottom_extend, left_extend, right_extend,
                                        cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Still in 8-bit gray format
        # 2020-2-29: Need to try if we can get it back to color image
        #            Otherwise the extracted meteor image seems to be
        #            in gray color as well
        # extend_img = cv2.cvtColor(extend_img, cv2.COLOR_BGR2GRAY)

        # Remove the position info and other info from the file name
        # Leave the detection # info (the 0003 in this file name example:
        #     ER4A3109_size(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
        #
        string_to_match = '_pos_('
        str_pos = image_file.find(string_to_match)

        if str_pos > -1:
            filename_no_ext = filename_no_ext[0:str_pos]

        file_to_save = filename_no_ext + file_ext
        file_to_save = os.path.join(save_dir, file_to_save)

        cv2.imwrite(file_to_save, extend_img)


# Deprecated !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# def extract_meteor_from_photo_file_with_mask_file(self, photo_dir, mask_dir, save_dir):
# photo_dir: The original photo folder
# mask_dir : The folder contains mask files which have been extended back
#            to original photo size
# NOTE:
# ==================================================================
# As processing the extraction with original photo size is quite
# slow. This is to be deprecated !!!
def extract_meteor_from_photo_folder_with_mask(self, photo_dir, mask_dir, save_dir, verbose):
    print("\nExtrating the meteor from original photo file. This would be quite slow ...")
    included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    mask_list = [fn for fn in os.listdir(mask_dir)
                 if any(fn.endswith(ext) for ext in included_extensions)]

    photo_list = [fn for fn in os.listdir(photo_dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    photo_list_no_ext = [os.path.splitext(fn)[0] for fn in photo_list]

    for mask_file in mask_list:
        mask_filename_no_ext, file_ext = os.path.splitext(mask_file)

        # Look for the corresponding original photo file
        # The mask file name would be like this:
        #     ER4A3035_size_(05760,03840)_0006.JPG
        #
        # Need to match with the first sub-string before the "_size_(" key word
        #
        # NO NO NO, no need to match the ext now. We'll use .png for those mask
        # and with the extension ".JPG"

        string_to_match = '_size_('
        str_pos = mask_filename_no_ext.find(string_to_match)

        if str_pos > -1:
            photo_file_name_no_ext = mask_filename_no_ext[0:str_pos]
            # photo_file_name += file_ext

            # if photo_file_name in photo_list:
            if photo_file_name_no_ext in photo_list_no_ext:
                list_index = photo_list_no_ext.index(photo_file_name_no_ext)

                # Seems no problem if the same file name with
                # different ext in the photo folder...
                photo_file_name = photo_list[list_index]

                photo_file_to_read = os.path.join(photo_dir, photo_file_name)
                mask_file_to_read = os.path.join(mask_dir, mask_file)

                photo_img = Image.open(photo_file_to_read)

                # Need to be processed in color
                # The mask image has been changed to 24-bit
                # photo_img = ImageOps.grayscale(photo_img)

                mask_img = Image.open(mask_file_to_read)

                if verbose:
                    print("... Extracting mask {} from photo {} ...".format(mask_file, photo_file_name))

                img_extract = ImageChops.multiply(photo_img, mask_img)
                img_extract = img_extract.convert("RGBA")
                datas = img_extract.getdata()

                newData = []
                for item in datas:
                    # if item[0] == 0 and item[1] == 0 and item[2] == 0:

                    # This still needs adjustment
                    # To remove some quite dark edge around the meteor
                    #
                    # Looks like currently 72 could be a reasonable
                    # threshold
                    #
                    # if item[0] < 8 and item[1] < 8 and item[2] < 8:
                    # if item[0] < 30 and item[1] < 30 and item[2] < 30:
                    # if item[0] < 80 and item[1] < 80 and item[2] < 80:
                    #
                    if item[0] < 72 and item[1] < 72 and item[2] < 72:
                        newData.append((255, 255, 255, 0))
                    else:
                        newData.append(item)

                img_extract.putdata(newData)

                # To be in PNG format
                file_to_save = mask_filename_no_ext + '_transparent.png'
                file_to_save = os.path.join(save_dir, file_to_save)
                img_extract.save(file_to_save, "PNG")
