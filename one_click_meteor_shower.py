import os
import sys

import detection
import gen_mask

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: one_click_meteor_shower.py <folder name>")
        sys.exit(1)

    original_dir = argv[0]
    if not os.path.exists(original_dir):
        print("No such directory: {}".format(original_dir))
        sys.exit(1)

    meteor_detector = detection.MeteorDetector()
    my_gen_mask = gen_mask.Gen_mask()

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '2_cropped' is hardcoded, don't change
    # Other sub-folder names in below can be changed
    extracted_dir = os.path.join(process_dir, '2_cropped')

    gray_256_dir = os.path.join(process_dir, '3_gray_256')
    mask_256_dir = os.path.join(process_dir, '4_mask_256')
    mask_resize_back_dir = os.path.join(process_dir, '5_mask_resize_back')
    # mask_extended_back_dir = os.path.join(process_dir, '6_mask_extended_back')
    object_extracted_dir = os.path.join(process_dir, '6_object_extracted')
    FINAL_dir = os.path.join(process_dir, '7_FINAL')
    FINAL_combined_dir = os.path.join(process_dir, '8_FINAL_combined')

    meteor_detector.detect_n_extract_meteor_from_folder(original_dir, process_dir, verbose=1)

    my_gen_mask.convert_image_folder_to_gray_256(extracted_dir, gray_256_dir)
    my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
    my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)
    # my_gen_mask.extend_mask_to_original_photo_size(mask_resize_back_dir, mask_extended_back_dir)
    # my_gen_mask.extract_meteor_from_photo_folder_with_mask(original_dir, mask_extended_back_dir, FINAL_dir, verbose=1)

    my_gen_mask.extract_meteor_from_cropped_folder_with_mask(extracted_dir,
                                                             mask_resize_back_dir,
                                                             object_extracted_dir,
                                                             verbose=1)
    my_gen_mask.extend_extracted_objects_to_original_photo_size(object_extracted_dir, FINAL_dir)
    my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)
