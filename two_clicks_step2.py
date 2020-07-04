# -*- coding: utf-8 -*-
import os
import sys

# import detection
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

    # meteor_detector = detection.MeteorDetector()
    my_gen_mask = gen_mask.Gen_mask()

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '02_cropped' is hardcoded, don't change
    # Other sub-folder names in below can be changed
    extracted_dir = os.path.join(process_dir, '02_cropped')

    filtered_dir = os.path.join(process_dir, '03_filtered')
    keep_dir = os.path.join(filtered_dir, 'good')
    # not_sure_dir = os.path.join(filtered_dir, 'not-sure')
    removed_dir = os.path.join(filtered_dir, 'removed')

    mosaic_dir = os.path.join(process_dir, '04_mosaic')
    gray_256_dir = os.path.join(process_dir, '05_gray_256')
    mask_256_dir = os.path.join(process_dir, '06_mask_256')
    mask_resize_back_dir = os.path.join(process_dir, '07_mask_resize_back')
    mosaic_merge_back_dir = os.path.join(process_dir, '08_mosaic_merged_back')

    # mask_extended_back_dir = os.path.join(process_dir, '6_mask_extended_back')
    object_extracted_dir = os.path.join(process_dir, '09_object_extracted')

    FINAL_dir = os.path.join(process_dir, '10_FINAL')
    FINAL_w_label_dir = os.path.join(process_dir, '10_FINAL_w_label')
    FINAL_combined_dir = os.path.join(process_dir, '11_FINAL_combined')

    my_gen_mask.convert_cropped_image_folder_to_mosaic_for_big_files(keep_dir, mosaic_dir)
    my_gen_mask.convert_image_folder_to_gray_256(mosaic_dir, gray_256_dir)
    my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
    my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)
    my_gen_mask.mosaic_mask_files_merge_back(mask_resize_back_dir, mosaic_merge_back_dir)
    my_gen_mask.extract_meteor_from_cropped_folder_with_mask(keep_dir,
                                                             mosaic_merge_back_dir,
                                                             object_extracted_dir,
                                                             verbose=1)
    # my_gen_mask.extend_extracted_objects_to_original_photo_size(object_extracted_dir, FINAL_dir)
    my_gen_mask.extend_extracted_objects_to_original_photo_size_by_multi_threading(object_extracted_dir,
                                                                                   FINAL_dir,
                                                                                   FINAL_w_label_dir)
    # my_gen_mask.print_filename_label_to_individual_final_image(FINAL_dir, FINAL_w_label_dir)
    my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, 'final.png', verbose=1)
    my_gen_mask.combine_meteor_images_to_one(FINAL_w_label_dir, FINAL_combined_dir, 'final_w_label.png', verbose=1)
    print("\nProcess finished!")
