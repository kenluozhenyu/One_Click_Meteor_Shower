# -*- coding: utf-8 -*-
import cv2
import os
import math
import shutil
import threading
import multiprocessing
from time import sleep
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFont

import model
import unet_proc

import settings


class Gen_mask:
    # Not all cropped images will be divided to mosaic
    # Only when images which width > 640 * 1.5
    #     settings.DETECTION_CROP_IMAGE_BOX_SIZE (640)
    #     settings.RATIO_FOR_MOSAIC (1.5)
    def convert_cropped_image_folder_to_mosaic_for_big_files(self, file_dir, save_dir):
        print("\nConverting the detected meteor images to mosaic if they are big ...")

        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']
        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        target_width = settings.DETECTION_CROP_IMAGE_BOX_SIZE

        for image_file in image_list:
            filename_w_path = os.path.join(file_dir, image_file)
            filename_no_ext, file_ext = os.path.splitext(image_file)

            original_img = cv2.imread(filename_w_path)

            orig_height = original_img.shape[0]
            orig_width = original_img.shape[1]

            # Normally this is >= 640 * 1.5
            if orig_width >= target_width * settings.RATIO_FOR_MOSAIC:
                # Big image, let's do mosaic
                #
                #    num_X ->
                #   -------------------
                #   |   |   |   |   | |
                #   |   |   |   |   | | num_Y |
                #   -------------------       V
                #   |   |   |   |   | |
                #   |   |   |   |   | |
                #   -------------------
                #   |   |   |   |   | |
                #   -------------------

                # round up, like 3.2 => 4

                # num_X = math.ceil(orig_width / target_width)
                # num_Y = math.ceil(orig_height / target_width)

                num_X_no_overlap = orig_width / target_width
                num_Y_no_overlap = orig_height / target_width

                overlap_ratio = settings.MOSAIC_OVERLAP_RATIO

                num_X_w_overlap = math.ceil((num_X_no_overlap - overlap_ratio) / (1 - overlap_ratio))
                num_Y_w_overlap = math.ceil((num_Y_no_overlap - overlap_ratio) / (1 - overlap_ratio))

                # for i in range(num_Y):
                for i in range(num_Y_w_overlap):
                    y1 = int(target_width * (i - overlap_ratio * i))
                    y2 = y1 + target_width

                    if y2 >= orig_height:
                        # y2 = orig_height - 1
                        y2 = orig_height - 1
                        y1 = orig_height - target_width

                    for j in range(num_X_w_overlap):
                        x1 = int(target_width * (j - overlap_ratio * j))
                        x2 = x1 + target_width

                        if x2 >= orig_width:
                            # x2 = orig_width - 1
                            x2 = orig_width
                            x1 = orig_width - target_width

                        mosaic_img = original_img[y1:y2, x1:x2]
                        file_to_save = filename_no_ext + \
                                       "_mosaic_({:03d},{:03d})_({:03d},{:03d})". \
                                           format(num_Y_w_overlap, num_X_w_overlap, i + 1, j + 1) + \
                                       file_ext

                        file_to_save = os.path.join(save_dir, file_to_save)
                        cv2.imwrite(file_to_save, mosaic_img)

            else:
                # No need to do mosaic
                # Just save the original image to the mosaic folder
                file_to_save = os.path.join(save_dir, image_file)
                cv2.imwrite(file_to_save, original_img)
            # sleep(0.02)
        # End for-loop

    def __convert_image_to_gray_256(self, original_img):
        # img = Image.open(file_to_open).convert('L')

        # img = color.rgb2gray(io.imread(file_to_open))

        # img = original_img.convert('L')

        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        basewidth = settings.UNET_IMAGE_SIZE

        # width_percent = (basewidth / float(img.size[0]))
        width_percent = (basewidth / gray.shape[0])
        height = int((float(gray.shape[1]) * float(width_percent)))

        # img = img.resize((basewidth, heigh), PIL.Image.ANTIALIAS)
        gray_256 = cv2.resize(gray, (basewidth, height))

        return gray_256

    # The orig_filename doesn't have path info. Just pure file name
    def __convert_image_file_to_gray_256(self, file_dir, orig_filename, save_dir):
        filename_w_path = os.path.join(file_dir, orig_filename)
        filename_no_ext, file_ext = os.path.splitext(orig_filename)

        img = cv2.imread(filename_w_path)
        gray_256 = self.__convert_image_to_gray_256(img)

        file_gray_256 = filename_no_ext + "_gray_256" + file_ext
        file_gray_256 = os.path.join(save_dir, file_gray_256)

        cv2.imwrite(file_gray_256, gray_256)

    def convert_image_folder_to_gray_256(self, file_dir, save_dir):
        print("\nConverting the detected meteor images to gray 256x256 size ...")

        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']
        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for image_file in image_list:
            self.__convert_image_file_to_gray_256(file_dir, image_file, save_dir)
            # sleep(0.2)

    # image_folder is the folder contains processed image, 256x256, gray
    def gen_meteor_mask_from_folder(self, image_folder, output_folder):
        print("\nGenerating mask from Unet ...")

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # The image size supported is (256, 256)
        unet_model = model.unet(input_size=(settings.UNET_IMAGE_SIZE, settings.UNET_IMAGE_SIZE, 1))
        unet_model.load_weights(settings.UNET_SAVED_MODEL)

        test_image_list = os.listdir(image_folder)
        num_image = len(test_image_list)

        testGene = unet_proc.testGenerator(image_folder, as_gray=True)

        '''
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        batch_size = 1

        # test_folder = os.path.dirname(image_folder)
        test_generator = test_datagen.flow_from_directory(
            image_folder,
            target_size=(256, 256),
            batch_size=batch_size,
            color_mode='grayscale',
            shuffle=False,
            class_mode=None)
        '''

        results = unet_model.predict_generator(testGene, num_image, verbose=1)
        # results = unet_model.predict_generator(test_generator, num_image, verbose=1)

        unet_proc.saveResult_V2(output_folder, results, test_image_list)

    # The filename parm doesn't have path info, just pure file name
    # The file name would be like this:
    #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
    #
    # The pos (x1, y1) (x2, y2) are the position from original image
    # Need to get this info back
    #
    def get_image_pos_from_file_name(self, filename):
        string_to_match = '_pos_('
        str_pos = filename.find(string_to_match)

        if str_pos == -1:
            return 0, 0, 0, 0

        str_x1 = filename[str_pos + 6:str_pos + 11]
        str_y1 = filename[str_pos + 12:str_pos + 17]

        str_x2 = filename[str_pos + 20:str_pos + 25]
        str_y2 = filename[str_pos + 26:str_pos + 31]

        x1 = int(str_x1)
        y1 = int(str_y1)
        x2 = int(str_x2)
        y2 = int(str_y2)

        return x1, y1, x2, y2

    # The filename parm doesn't have path info, just pure file name
    # The file name would be like this:
    #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
    #
    # The (x1, y1) (x2, y2) are the position from original image
    # Need to get this info back
    #
    def get_image_size_from_file_name(self, filename):
        string_to_match = '_size_('
        str_pos = filename.find(string_to_match)

        if str_pos == -1:
            return 0, 0

        str_x = filename[str_pos + 7:str_pos + 12]
        str_y = filename[str_pos + 13:str_pos + 18]

        x = int(str_x)
        y = int(str_y)

        return x, y

    # This is to resize the 256x256 mask file back to its
    # original cropped XXX x XXX size. Still a small image
    # Not the original photo file size
    def resize_mask_to_original_cropped_size(self, file_dir, save_dir):
        print("\nResizing the mask back to original cropped size ...")

        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']
        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for image_file in image_list:
            filename_w_path = os.path.join(file_dir, image_file)
            filename_no_ext, file_ext = os.path.splitext(image_file)

            # The file name would be like this:
            #     ER4A3035_0003_pos_(01653,02734)_(02293,03374)_gray_256_mask.JPG
            # The (x1, y1) (x2, y2) are the position from original image
            # Need to get this info back
            #
            # If the file comes from a mosaic, it would be like this:
            # IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988)_mosaic_(002,002)_(001,001)_gray_256_mask_1050.png
            # Normally it needs to be re-sized back to settings.DETECTION_CROP_IMAGE_BOX_SIZE (640)
            string_to_match = '_mosaic_('
            str_pos_mosaic = image_file.find(string_to_match)

            if str_pos_mosaic == -1:
                # Normal file, read the original position info to
                # determine the size

                x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)

                img = cv2.imread(filename_w_path)

                # original_width = settings.detection_crop_img_box_size
                original_width = abs(x2 - x1)

                # width_percent = (basewidth / float(img.size[0]))
                width_percent = (original_width / img.shape[0])
                height = int((float(img.shape[1]) * float(width_percent)))
            else:
                # Mosaic image, the original size should be
                # settings.DETECTION_CROP_IMAGE_BOX_SIZE (normally it is 640)
                original_width = settings.DETECTION_CROP_IMAGE_BOX_SIZE
                height = settings.DETECTION_CROP_IMAGE_BOX_SIZE
                img = cv2.imread(filename_w_path)

            resized_img = cv2.resize(img, (original_width, height))

            # Still in 8-bit gray format
            #
            # 2020-3-14:
            # NO. This will be used for extracted the meteor object directly
            # Keep it as 24-bit
            # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            file_to_save = filename_no_ext + "_{}".format(original_width) + file_ext
            file_to_save = os.path.join(save_dir, file_to_save)

            cv2.imwrite(file_to_save, resized_img)
            # sleep(0.02)

    # After the masks are generated, and re-sized back (normally 640x640),
    # scan the ones from mosaic, and merge them back to one file
    # For images don't belong to mosaic, just save them to the new folder
    def mosaic_mask_files_merge_back(self, file_dir, save_dir):
        print("\nMerging the mosaic images back to one file ...")

        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']
        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Need to be sorted in Ascending order by the file name
        image_list.sort()

        image_list_iter = iter(enumerate(image_list))

        for index, image_file in image_list_iter:
            # The file name would be like this:
            # IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988)_mosaic_(002,002)_(001,001)_gray_256_mask_1050.png
            string_to_match = '_mosaic_('
            str_pos_mosaic = image_file.find(string_to_match)

            if str_pos_mosaic > -1:
                # Ok this file should be the first one
                # of a group of mosaic images

                # To determine the original mosaic size,
                # need to get he position info
                string_to_match = '_pos_('
                str_pos = image_file.find(string_to_match)

                if str_pos == -1:
                    continue

                '''
                str_x1 = image_file[str_pos + 6:str_pos + 11]
                str_y1 = image_file[str_pos + 12:str_pos + 17]

                str_x2 = image_file[str_pos + 20:str_pos + 25]
                str_y2 = image_file[str_pos + 26:str_pos + 31]

                x1 = int(str_x1)
                y1 = int(str_y1)
                x2 = int(str_x2)
                y2 = int(str_y2)
                '''

                x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)

                # Normally these two values should be the same
                orig_width = abs(x2 - x1)
                orig_height = abs(y2 - y1)

                # Get the X/Y number of pictures
                str_y = image_file[str_pos_mosaic + 9:str_pos_mosaic + 12]
                str_x = image_file[str_pos_mosaic + 13:str_pos_mosaic + 16]

                x = int(str_x)
                y = int(str_y)

                # num_of_pics_for_mosaic = x * y

                # Including the current image, totally num_of_pics_for_mosaic
                # should be merged
                # index range == i, i+1, i+2 ... i+num-1

                # file_to_read = image_file
                # file_to_read = os.path.join(image_dir, file_to_read)
                # img_mosaic = cv2.imread(file_to_read)
                # img_mosaic = None

                img_mosaic = Image.new('RGB', (orig_width, orig_height))

                overlap_ratio = settings.MOSAIC_OVERLAP_RATIO

                icount = 0
                for i in range(y):
                    # y_paste_from =
                    for j in range(x):
                        file_to_read = image_list[index + icount]
                        file_to_read = os.path.join(file_dir, file_to_read)
                        # img = cv2.imread(file_to_read)
                        img = Image.open(file_to_read)

                        # actually this should be the same
                        img_width = img.width

                        y_paste_from = int(img_width * (i - overlap_ratio * i))
                        if (y_paste_from + img_width) >= orig_height:
                            y_paste_from = orig_height - img_width

                        x_paste_from = int(img_width * (j - overlap_ratio * j))
                        if (x_paste_from + img_width) >= orig_width:
                            x_paste_from = orig_width - img_width

                        # To ensure the generated mask (RGB(255,255,255)) part is not overridden,
                        # do a logical_or operation with the existing part
                        mask = img_mosaic.crop((x_paste_from,
                                                y_paste_from,
                                                x_paste_from + img_width,
                                                y_paste_from + img_width)).convert("1")
                        img = ImageChops.logical_or(img.convert("1"), mask)

                        img_mosaic.paste(img, (x_paste_from, y_paste_from))
                        icount += 1

                # The file name would be like this:
                # From:
                # IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988)_mosaic_(002,002)_(001,001)_gray_256_mask_1050.png
                # To:
                # IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988)_gray_256_mask_1050.png
                #
                file_to_save = image_file[0: str_pos_mosaic] + image_file[str_pos_mosaic + 27: len(image_file)]
                # file_to_save = image_file[0: str_pos_mosaic] + "_gray_mask"
                file_to_save = os.path.join(save_dir, file_to_save)
                img_mosaic.save(file_to_save, 'PNG')

                # Done for this mosaic
                # Need to skip (x*y-1) items
                for i in range(x * y - 1):
                    next(image_list_iter)

            else: # if str_pos_mosaic > -1:
                # It should be a normal image
                # save it to the new folder
                orig_file = image_file
                orig_file = os.path.join(file_dir, orig_file)
                file_to_save = image_file
                file_to_save = os.path.join(save_dir, file_to_save)
                shutil.copyfile(orig_file, file_to_save)
            # sleep(0.02)
        # end of for-loop

    def extract_meteor_from_cropped_file_with_mask(self, cropped_photo_file, mask_file, save_file):
        cropped_img = Image.open(cropped_photo_file)

        # Need to be processed in color
        # The mask image has been changed to 24-bit
        # photo_img = ImageOps.grayscale(photo_img)

        mask_img = Image.open(mask_file)

        img_extract = ImageChops.multiply(cropped_img, mask_img)
        img_extract = img_extract.convert("RGBA")
        datas = img_extract.getdata()

        newData = []
        rgb_threshold = settings.EXTRACT_RGB_VALUE_THRESHOLD
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
            # if item[0] < 72 and item[1] < 72 and item[2] < 72:
            # if item[0] < 30 and item[1] < 30 and item[2] < 30:
            if item[0] < rgb_threshold and item[1] < rgb_threshold and item[2] < rgb_threshold:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img_extract.putdata(newData)

        # To be in PNG format
        # file_to_save = mask_filename_no_ext + '_transparent.png'
        # file_to_save = os.path.join(save_dir, file_to_save)
        img_extract.save(save_file, "PNG")

    # def extract_meteor_from_cropped_folder_with_mask(self, photo_dir, mask_dir, save_dir):
    # crop_dir: The cropped meteor objects photo folder. Normally it is the "2_extraction"
    # mask_dir : The folder contains mask files which have been extended back
    #            to original photo size
    def extract_meteor_from_cropped_folder_with_mask(self, cropped_dir, mask_dir, save_dir, verbose):
        print("\nExtrating the meteor from cropped files...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        mask_list = [fn for fn in os.listdir(mask_dir)
                     if any(fn.endswith(ext) for ext in included_extensions)]

        cropped_list = [fn for fn in os.listdir(cropped_dir)
                        if any(fn.endswith(ext) for ext in included_extensions)]

        cropped_list_no_ext = [os.path.splitext(fn)[0] for fn in cropped_list]

        for mask_file in mask_list:
            mask_filename_no_ext, file_ext = os.path.splitext(mask_file)

            # To be in PNG format
            file_to_save = mask_filename_no_ext + '_transparent.png'
            file_to_save = os.path.join(save_dir, file_to_save)

            # Look for the corresponding original photo file
            # The mask file name and the cropped image file name would be like these:
            #     IMG_3077_size_(05472,03648)_0001_pos_(01567,00746)_(02207,01386)_gray_256_mask_640.png
            #     IMG_3077_size_(05472,03648)_0001_pos_(01567,00746)_(02207,01386).JPG
            # Need to match with the first sub-string before the "_gray_256_mask_"
            # key word
            #
            # NO NO NO, no need to match the ext now. We'll use .png for those mask

            string_to_match = '_gray_256_mask_'
            str_pos = mask_filename_no_ext.find(string_to_match)

            if str_pos > -1:
                cropped_file_name_no_ext = mask_filename_no_ext[0:str_pos]
                # photo_file_name += file_ext

                # if photo_file_name in photo_list:
                if cropped_file_name_no_ext in cropped_list_no_ext:
                    list_index = cropped_list_no_ext.index(cropped_file_name_no_ext)

                    # Seems no problem if the same file name with
                    # different ext in the photo folder...
                    cropped_file_name = cropped_list[list_index]

                    cropped_file_to_read = os.path.join(cropped_dir, cropped_file_name)
                    mask_file_to_read = os.path.join(mask_dir, mask_file)

                    if verbose:
                        print("... Extracting mask {} from cropped photo {} ...".format(mask_file, cropped_file_name))

                    self.extract_meteor_from_cropped_file_with_mask(cropped_file_to_read,
                                                                    mask_file_to_read,
                                                                    file_to_save)
        # end for loop of the mask_list

    def extract_meteor_from_original_file_with_mask(self, original_photo_file, mask_file, save_file):
        original_img = Image.open(original_photo_file)

        # Need to get the position info from the mask file name
        x1, y1, x2, y2 = self.get_image_pos_from_file_name(mask_file)
        # cropped_img = original_img[y1:y2, x1:x2]
        cropped_img = original_img.crop((x1, y1, x2, y2))

        # Need to be processed in color
        # The mask image has been changed to 24-bit
        # photo_img = ImageOps.grayscale(photo_img)

        mask_img = Image.open(mask_file)

        img_extract = ImageChops.multiply(cropped_img, mask_img)
        img_extract = img_extract.convert("RGBA")
        datas = img_extract.getdata()

        newData = []
        rgb_threshold = settings.EXTRACT_RGB_VALUE_THRESHOLD
        for item in datas:
            if item[0] < rgb_threshold and item[1] < rgb_threshold and item[2] < rgb_threshold:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img_extract.putdata(newData)

        # To be in PNG format
        # file_to_save = mask_filename_no_ext + '_transparent.png'
        # file_to_save = os.path.join(save_dir, file_to_save)
        img_extract.save(save_file, "PNG")

    # We may allow some process on the cropped image, like improving the contrast to make it better
    # to be process by the UNET network.
    # In that case when doing the final extraction, we'd better to extract the meteor object from
    # the original images.
    def extract_meteor_from_original_folder_with_mask(self, original_dir, mask_dir, save_dir, verbose):
        print("\nExtrating the meteor from cropped files...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        mask_list = [fn for fn in os.listdir(mask_dir)
                     if any(fn.endswith(ext) for ext in included_extensions)]

        original_list = [fn for fn in os.listdir(original_dir)
                        if any(fn.endswith(ext) for ext in included_extensions)]

        original_list_no_ext = [os.path.splitext(fn)[0] for fn in original_list]

        for mask_file in mask_list:
            mask_filename_no_ext, file_ext = os.path.splitext(mask_file)

            # To be in PNG format
            file_to_save = mask_filename_no_ext + '_transparent.png'
            file_to_save = os.path.join(save_dir, file_to_save)

            # Look for the corresponding original photo file
            # The mask file name and the original image file name would be like these:
            #     IMG_3077_size_(05472,03648)_0001_pos_(01567,00746)_(02207,01386)_gray_256_mask_640.png
            #     IMG_3077_size_(05472,03648)_0001_pos_(01567,00746)_(02207,01386).JPG
            # Need to match with the first sub-string before the "_size_("
            # key word
            #
            # NO NO NO, no need to match the ext now. We'll use .png for those mask

            string_to_match = '_size_('
            str_pos = mask_filename_no_ext.find(string_to_match)

            if str_pos > -1:
                original_file_name_no_ext = mask_filename_no_ext[0:str_pos]
                # photo_file_name += file_ext

                # if photo_file_name in photo_list:
                if original_file_name_no_ext in original_list_no_ext:
                    list_index = original_list_no_ext.index(original_file_name_no_ext)

                    # Seems no problem if the same file name with
                    # different ext in the photo folder...
                    original_file_name = original_list[list_index]

                    original_file_to_read = os.path.join(original_dir, original_file_name)
                    mask_file_to_read = os.path.join(mask_dir, mask_file)

                    if verbose:
                        print("... Extracting mask {} from original photo {} ...".format(mask_file, original_file_name))

                    self.extract_meteor_from_original_file_with_mask(original_file_to_read,
                                                                     mask_file_to_read,
                                                                     file_to_save)
            # sleep(0.02)
        # end for loop of the mask_list

    # This extends the XXX x XXX extracted meteor objects png file to the original big
    # photo file size. Still in png format
    #
    # The file name is like this:
    #     IMG_3039_size_(05472,03648)_0006_pos_(02264,00000)_(02904,00640)_gray_256_mask_640_transparent.png
    #
    # This process is a little bit slow. Better to make it multi-threaded
    # -- This is implemented by calling the
    #        extend_extracted_objects_to_original_photo_size_by_multi_threading
    #    function
    # When using multi-thread mode, the "selected_image_list" parameter is to
    # be used. It specify a sub-set of the image list to be handled by a thread.
    def extend_extracted_objects_to_original_photo_size(self, file_dir, save_dir, label_save_dir,
                                                        selected_image_list=[], verbose=1):
        if len(selected_image_list) == 0:
            print("\nExtending the extracted objects back to original photo size ...")
            # No file list specified
            # Get all image file list in the folder
            included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff',
                                   'TIFF']

            image_list = [fn for fn in os.listdir(file_dir)
                          if any(fn.endswith(ext) for ext in included_extensions)]
        else:
            image_list = selected_image_list

        try:
            ttFont = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            ttFont = ImageFont.load_default()

        for image_file in image_list:
            if verbose:
                print("... Processing {} ...".format(image_file))
            filename_w_path = os.path.join(file_dir, image_file)
            filename_no_ext, file_ext = os.path.splitext(image_file)

            # The file name would be like this:
            #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
            # The size (x, y) is the original photo size
            # The pos (x1, y1) (x2, y2) are the position from original image
            # Need to get these info back
            x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)
            target_width, target_height = self.get_image_size_from_file_name(image_file)

            # To load a PNG image with 4 channels in OpenCV,
            # use im = cv2.imread(file, cv2.IMREAD_UNCHANGED).
            # You will obtain a BGRA image.
            # img = cv2.imread(filename_w_path)
            img = cv2.imread(filename_w_path, cv2.IMREAD_UNCHANGED)

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
            # Leave the detection # info (the 0001 in this file name example:
            #     ER4A3109_r_size_(05760,03840)_0001_pos_(02301,02327)_(02941,02967)_gray_256_mask_640_transparent.png
            #
            # 2020-7-4: Decided to leave the position info in he file name.
            #           Because we may want to have a copy to print the file
            #           label near the meteor object.

            # string_to_match = '_pos_('
            string_to_match = '_gray_256_mask_'

            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                filename_no_ext = filename_no_ext[0:str_pos]

            filename_to_save = filename_no_ext + file_ext
            file_to_save = os.path.join(save_dir, filename_to_save)

            cv2.imwrite(file_to_save, extend_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            # 2020-7-4:
            # Add the file name as the label to the image, and save to another location
            string_to_match = '_center_('
            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                str_x_c = image_file[str_pos + 9:str_pos + 14]
                str_y_c = image_file[str_pos + 15:str_pos + 20]

                x_c = int(str_x_c)
                y_c = int(str_y_c) - 16
            else:
                x_c = 0
                y_c = 0

            # Get the short file name.
            # The file name would be:
            #     ER4A3109_r_size_(05760,03840)_0001_pos_(02301,02327)_(02941,02967).png
            #
            # The short file name would be:
            #     ER4A3109_r_0001
            label_name = image_file
            string_to_match = '_pos_('
            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                label_name = image_file[0:str_pos - 24] + image_file[str_pos - 5:str_pos]

            im_rgb = cv2.cvtColor(extend_img, cv2.COLOR_BGRA2RGBA)
            pil_im = Image.fromarray(im_rgb)

            draw = ImageDraw.Draw(pil_im)
            draw.text((x_c, y_c), label_name, fill=(0, 255, 255), font=ttFont)

            # b, g, r = pil_im.split()
            # pil_im = Image.merge("RGB", (r, g, b))

            '''
            cv2.putText(extend_img, label_name,
                        (x_c, y_c),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3,
                        color=(0, 255, 255),
                        lineType=2)
            '''
            file_to_save = os.path.join(label_save_dir, filename_to_save)

            pil_im.save(file_to_save, "PNG")
            # cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(file_to_save, cv2_im_processed, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            # sleep(0.02)
        # end for loop

    def extend_extracted_objects_to_original_photo_size_by_multi_threading(self, file_dir, save_dir, label_save_dir,
                                                                           verbose=1):
        print("\nExtending the extracted objects back to original photo size ...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff',
                               'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        if not os.path.exists(label_save_dir):
            os.mkdir(label_save_dir)

        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        CPU_count = multiprocessing.cpu_count()

        # Add some restriction to avoid out of memory
        # if CPU_count > 8:
        #     CPU_count = 8
        if CPU_count > settings.MAX_CPU_FOR_MASK_EXTRACTION:
            CPU_count = settings.MAX_CPU_FOR_MASK_EXTRACTION

        num_image_list = len(image_list)

        size_per_sublist = math.ceil(num_image_list / CPU_count)
        print('    Totally {} images to be processed by {} CPU cores'.format(num_image_list, CPU_count))
        print("    Each core to handle {} images".format(size_per_sublist))

        thread_set = []

        start_from = 0
        num = 0

        # i = 0, 1, ... CPU_count - 1
        for i in range(CPU_count):
            # print(len(url_list_set[i]))
            # print(url_list_set[i])

            # 0 ~ 9:   10
            # 10 ~ 19: 10
            start_from = size_per_sublist * i

            # Extreme case, the CPU count is larger than
            # the number of images, we don't need to
            # create new thread
            if start_from >= num_image_list:
                break

            num = size_per_sublist
            if start_from + num > num_image_list:
                # (num_image_list-1) is the maximum index of the list
                num = (num_image_list-1)-start_from+1

            # print('\nThread-{0:03d}:'.format(i))
            # print(start_from)
            # print(num)

            subset_image_list = image_list[start_from:start_from+num]
            # print(subset_image_list)

            thread_set.append(threading.Thread(target=self.extend_extracted_objects_to_original_photo_size,
                                               args=(file_dir, save_dir, label_save_dir, subset_image_list, verbose)))

        # thread_set = [myThread(i + 1, "Thread-{0:03d}".format(i + 1), start_from, num) for i in range(NUM_OF_THREADS)]

        for index, thread_process in enumerate(thread_set):
            thread_process.start()
            print('    Thread # {0:03d} started ...'.format(index))

        for thread_process in thread_set:
            thread_process.join()

        print("\nMulti-thread process done !")

    # Sometimes the final combined image would still contain some objects we don't want.
    # Like satellites (escaped from recognition), or a few meteors we don't want.
    #
    # Print the file label near to the object could help to easily identify which file
    # we want to exclude from the final combination
    #
    # 2020-7-4: This is no need to use now. The function is combined to
    #           self.extend_extracted_objects_to_original_photo_size
    def print_filename_label_to_individual_final_image(self, file_dir, save_dir, verbose=1):
        print("\nGenerating files with label ...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff',
                               'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

        try:
            ttFont = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            ttFont = ImageFont.load_default()

        for image_file in image_list:
            if verbose:
                print("... Processing {} ...".format(image_file))

            filename_w_path = os.path.join(file_dir, image_file)

            # x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)
            # x0 = int((x1 + x2) / 2)
            # y0 = int((y1 + y2) / 2) - 20

            string_to_match = '_center_('
            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                str_x_c = image_file[str_pos + 9:str_pos + 14]
                str_y_c = image_file[str_pos + 15:str_pos + 20]

                x_c = int(str_x_c)
                y_c = int(str_y_c) - 16
            else:
                x_c = 0
                y_c = 0

            im = Image.open(filename_w_path)
            draw = ImageDraw.Draw(im)

            # Get the short file name.
            # The file name would be:
            #     ER4A3109_r_size_(05760,03840)_0001_pos_(02301,02327)_(02941,02967).png
            #
            # The short file name would be:
            #     ER4A3109_r_size_0001

            label_name = image_file

            string_to_match = '_pos_('
            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                label_name = image_file[0:str_pos-24] + image_file[str_pos-5:str_pos]

            draw.text((x_c, y_c), label_name, fill=(0, 255, 255), font=ttFont)

            file_to_save = os.path.join(save_dir, image_file)
            im.save(file_to_save, "PNG")
        # end for loop

    def combine_meteor_images_to_one(self, meteor_dir, save_dir, specified_filename='final.png', verbose=1):
        print("\nCombining the meteor images to {} ...".format(specified_filename))
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        meteor_list = [fn for fn in os.listdir(meteor_dir)
                       if any(fn.endswith(ext) for ext in included_extensions)]

        if len(meteor_list) == 0:
            print("No image file in folder {}".format(meteor_dir))
            return

        meteor_file = meteor_list[0]
        filename_w_path = os.path.join(meteor_dir, meteor_file)
        if verbose:
            print("... Merging {} ...".format(meteor_file))
        combined_img = Image.open(filename_w_path)

        i = 0

        for meteor_file in meteor_list:
            # The first image had been opened before the for-loop
            if i > 0:
                filename_w_path = os.path.join(meteor_dir, meteor_file)

                if verbose:
                    print("... Merging {} ...".format(meteor_file))

                    img = Image.open(filename_w_path)
                    combined_img = Image.alpha_composite(combined_img, img)
            i += 1
            # sleep(0.02)

        file_to_save = specified_filename
        file_to_save = os.path.join(save_dir, file_to_save)

        combined_img.save(file_to_save, 'PNG')
