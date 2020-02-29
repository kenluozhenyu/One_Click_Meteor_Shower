import cv2
import os
from PIL import Image, ImageOps, ImageChops

import model
import unet_proc

import settings


class Gen_mask:
    def __convert_image_to_gray_256(self, original_img):
        # img = Image.open(file_to_open).convert('L')

        # img = color.rgb2gray(io.imread(file_to_open))

        # img = original_img.convert('L')

        gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        basewidth = settings.unet_image_size

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
        unet_model = model.unet(input_size=(settings.unet_image_size, settings.unet_image_size, 1))
        unet_model.load_weights(settings.unet_saved_model)

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
            x1, y1, x2, y2 = self.get_image_pos_from_file_name(image_file)

            img = cv2.imread(filename_w_path)

            # original_width = settings.detection_crop_img_box_size
            original_width = abs(x2 - x1)

            # width_percent = (basewidth / float(img.size[0]))
            width_percent = (original_width / img.shape[0])
            height = int((float(img.shape[1]) * float(width_percent)))

            resized_img = cv2.resize(img, (original_width, height))

            # Still in 8-bit gray format
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            file_to_save = filename_no_ext + "_{}".format(original_width) + file_ext
            file_to_save = os.path.join(save_dir, file_to_save)

            cv2.imwrite(file_to_save, resized_img)

    # This extends the XXX x XXX cropped mask image to the original big
    # photo file size.
    #
    # The file name is like this:
    #     ER4A3109_size_(05437,03625)_0001_pos_(02194,02421)_(02834,03061).JPG
    #
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

    # def extract_meteor_from_photo_file_with_mask_file(self, photo_dir, mask_dir, save_dir):
    # photo_dir: The original photo folder
    # mask_dir : The folder contains mask files which have been extended back
    #            to original photo size
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
                        # Values like 080805, 0d0c08, 322d28, 0f0f0b, 080906...
                        if item[0] < 8 and item[1] < 8 and item[2] < 8:
                            newData.append((255, 255, 255, 0))
                        else:
                            newData.append(item)

                    img_extract.putdata(newData)

                    # To be in PNG format
                    file_to_save = mask_filename_no_ext + '_transparent.png'
                    file_to_save = os.path.join(save_dir, file_to_save)
                    img_extract.save(file_to_save, "PNG")

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

    original_dir = 'F:/meteor-one-click-test/'

    extracted_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/2_extraction/'
    gray_256_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/3_gray_256'
    mask_256_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/4_mask_256'

    # mask_resize_back_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/5_mask_resize_back'
    mask_resize_back_dir = 'F:/meteor-one-click-test/process/5_mask_resize_back'
    # mask_extended_back_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/6_mask_extended_back'
    mask_extended_back_dir = 'F:/meteor-one-click-test/process/6_mask_extended_back'

    # FINAL_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/7_FINAL'
    FINAL_dir = 'F:/meteor-one-click-test/process/7_FINAL'
    # FINAL_combined_dir = 'data/meteor/meteor-data/meteor-detection/original_images/extracted/8_FINAL_combined'
    FINAL_combined_dir = 'F:/meteor-one-click-test/process/8_FINAL_combined'

    my_gen_mask = Gen_mask()

    # my_gen_mask.convert_image_folder_to_gray_256(extracted_dir, gray_256_dir)
    # my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
    # my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)
    # my_gen_mask.extend_mask_to_original_photo_size(mask_resize_back_dir, mask_extended_back_dir)
    my_gen_mask.extract_meteor_from_photo_folder_with_mask(original_dir, mask_extended_back_dir, FINAL_dir, verbose=1)
    my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)
