# -*- coding: utf-8 -*-
import cv2
import os
import math
import shutil
from PIL import Image, ImageOps, ImageChops

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
        results = unet_model.predict_generator(testGene, num_image, verbose=1)

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
                file_to_save = image_file[0: str_pos_mosaic] + image_file[str_pos_mosaic + 27: len(image_file)]
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
        # end of for-loop

    def extract_meteor_from_file_with_mask(self, cropped_photo_file, mask_file, save_file):
        cropped_img = Image.open(cropped_photo_file)

        # Need to be processed in color
        # The mask image has been changed to 24-bit
        # photo_img = ImageOps.grayscale(photo_img)

        mask_img = Image.open(mask_file)

        img_extract = ImageChops.multiply(cropped_img, mask_img)
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
        # file_to_save = mask_filename_no_ext + '_transparent.png'
        # file_to_save = os.path.join(save_dir, file_to_save)
        img_extract.save(save_file, "PNG")

    # def extract_meteor_from_photo_file_with_mask_file(self, photo_dir, mask_dir, save_dir):
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

                    self.extract_meteor_from_file_with_mask(cropped_file_to_read,
                                                            mask_file_to_read,
                                                            file_to_save)
        # end for loop of the mask_list

    # This extends the XXX x XXX extracted meteor objects png file to the original big
    # photo file size. Still in png format
    #
    # The file name is like this:
    #     IMG_3039_size_(05472,03648)_0006_pos_(02264,00000)_(02904,00640)_gray_256_mask_640_transparent.png
    #
    # This process is a little bit slow. Better to make it multi-threaded
    def extend_extracted_objects_to_original_photo_size(self, file_dir, save_dir, verbose=1):
        print("\nExtending the extracted objects back to original photo size ...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff',
                               'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        image_list = [fn for fn in os.listdir(file_dir)
                      if any(fn.endswith(ext) for ext in included_extensions)]

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
            # Leave the detection # info (the 0003 in this file name example:
            #     ER4A3109_size(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
            #
            string_to_match = '_pos_('
            str_pos = image_file.find(string_to_match)

            if str_pos > -1:
                filename_no_ext = filename_no_ext[0:str_pos]

            file_to_save = filename_no_ext + file_ext
            file_to_save = os.path.join(save_dir, file_to_save)

            cv2.imwrite(file_to_save, extend_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    def combine_meteor_images_to_one(self, meteor_dir, save_dir, verbose):
        print("\nCombining the meteor images to one ...")
        included_extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'bmp', 'BMP', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF']

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        meteor_list = [fn for fn in os.listdir(meteor_dir)
                       if any(fn.endswith(ext) for ext in included_extensions)]

        if len(meteor_list) == 0:
            return

        meteor_file = meteor_list[0]
        filename_w_path = os.path.join(meteor_dir, meteor_file)
        if verbose:
            print("... Merging {} ...".format(meteor_file))
        combined_img = Image.open(filename_w_path)

        i = 0

        for meteor_file in meteor_list:
            if i > 0:
                filename_w_path = os.path.join(meteor_dir, meteor_file)

                if verbose:
                    print("... Merging {} ...".format(meteor_file))

                    img = Image.open(filename_w_path)
                    combined_img = Image.alpha_composite(combined_img, img)
            i += 1

        file_to_save = 'final.png'
        file_to_save = os.path.join(save_dir, file_to_save)

        combined_img.save(file_to_save, 'PNG')


if __name__ == "__main__":
    # original_dir = 'data/meteor/meteor-data/meteor-detection/original_images/'

    original_dir = 'F:/meteor-one-click-2018-08-milkyway'

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '2_cropped' is hardcoded, don't change
    # Other sub-folder names in below can be changed
    extracted_dir = os.path.join(process_dir, '02_cropped')

    mosaic_dir = os.path.join(process_dir, '03_mosaic')
    gray_256_dir = os.path.join(process_dir, '04_gray_256')
    mask_256_dir = os.path.join(process_dir, '05_mask_256')
    mask_resize_back_dir = os.path.join(process_dir, '06_mask_resize_back')
    mosaic_merge_back_dir = os.path.join(process_dir, '07_mosaic_merged_back')

    # mask_extended_back_dir = os.path.join(process_dir, '6_mask_extended_back')
    object_extracted_dir = os.path.join(process_dir, '08_object_extracted')

    FINAL_dir = os.path.join(process_dir, '09_FINAL')
    FINAL_combined_dir = os.path.join(process_dir, '10_FINAL_combined')

    my_gen_mask = Gen_mask()

    my_gen_mask.convert_cropped_image_folder_to_mosaic_for_big_files(extracted_dir, mosaic_dir)
    my_gen_mask.convert_image_folder_to_gray_256(mosaic_dir, gray_256_dir)
    my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
    my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)
    my_gen_mask.mosaic_mask_files_merge_back(mask_resize_back_dir, mosaic_merge_back_dir)
    my_gen_mask.extract_meteor_from_cropped_folder_with_mask(extracted_dir,
                                                             mosaic_merge_back_dir,
                                                             object_extracted_dir,
                                                             verbose=1)

    # my_gen_mask.extract_meteor_from_photo_folder_with_mask(original_dir, mask_extended_back_dir, FINAL_dir, verbose=1)

    my_gen_mask.extend_extracted_objects_to_original_photo_size(object_extracted_dir, FINAL_dir)
    my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)

    # my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)

    # cropped_photo_file = 'F:/test/IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988).JPG'
    # mask_file = 'F:/test/IMG_3119_size_(05472,03648)_0001_pos_(02650,01938)_(03700,02988)_gray_256_mask_1050.png'
    # save_file = 'F:/test/IMG_3119_extracted_v2.png'
    # my_gen_mask.extract_meteor_from_file_with_mask(cropped_photo_file, mask_file, save_file)
