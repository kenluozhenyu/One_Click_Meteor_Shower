import os
import sys

import detection
import gen_mask

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        print("\nUsage: auto_meteor_shower <operation> <equatorial_mount option: Y/N> <folder name>")
        print("operation: all (Do for both detection and extraction)")
        print("           detection  (This is the step 1. Detection only)")
        print("           extraction (This is the step 2. Extraction only)")
        print("                      (The equatorial_mount option is not needed for the extraction-only operation)")
        print("\nequatorial_mount option: Y (If the images were taken on equatorial mount)")
        print("                         N (Choose this for images taken on fixed tripod)")
        sys.exit(1)

    do_option = argv[0]
    do_option = do_option.lower()
    if do_option != "all" and do_option != "detection" and do_option != "extraction":
        print("\nUsage: auto_meteor_shower <operation> <equatorial_mount option: Y/N> <folder name>")
        print("operation: all (Do for both detection and extraction)")
        print("           detection  (This is the step 1. Detection only)")
        print("           extraction (This is the step 2. Extraction only)")
        print("                      (The equatorial_mount option is not needed for the extraction-only operation)")
        print("\nequatorial_mount option: Y (If the images were taken on equatorial mount)")
        print("                         N (Choose this for images taken on fixed tripod)")
        sys.exit(1)

    equatorial_mount_option = "N"

    if do_option == "all" or do_option != "detection":
        equatorial_mount_option = argv[1]
        equatorial_mount_option = equatorial_mount_option.upper()
        if equatorial_mount_option != "Y" and equatorial_mount_option != "N":
            if do_option != "extraction":
                print("\nUsage: auto_meteor_shower <operation> <equatorial_mount option: Y/N> <folder name>")
                print("operation: all (Do for both detection and extraction)")
                print("           detection  (This is the step 1. Detection only)")
                print("           extraction (This is the step 2. Extraction only)")
                print("                      (The equatorial_mount option is not needed for the extraction-only operation)")
                print("\nequatorial_mount option: Y (If the images were taken on equatorial mount)")
                print("                         N (Choose this for images taken on fixed tripod)")
                sys.exit(1)

        original_dir = argv[2]
    else:
        original_dir = argv[1]

    if not os.path.exists(original_dir):
        print("\nNo such directory: {}".format(original_dir))
        sys.exit(1)

    meteor_detector = detection.MeteorDetector()
    my_gen_mask = gen_mask.Gen_mask()

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '2_cropped' is hardcoded, don't change
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

    if do_option == "all" or do_option == "detection":
        is_equatorial_mount = False
        if equatorial_mount_option == 'Y':
            is_equatorial_mount = True

        detection.multi_thread_process_detect_n_extract_meteor_from_folder(original_dir,
                                                                           process_dir,
                                                                           subtraction=True,
                                                                           equatorial_mount=is_equatorial_mount,
                                                                           verbose=1)

        # meteor_detector.filter_possible_not_meteor_objects(extracted_dir, keep_dir, not_sure_dir, removed_dir)
        # meteor_detector.filter_possible_not_meteor_objects(extracted_dir, keep_dir, removed_dir)
        detection.filter_possible_not_meteor_objects(extracted_dir, keep_dir, removed_dir)

    if do_option == "detection":
        print(
            "\n======================================================================================================")
        print("\nPossible objects extraction finished.")
        # print("You may go to the "'02_cropped'" folder to double check the detection objects.")
        print("\nYou may go to the "'03_filtered'" folder to double check the detection objects.")
        print("The 'good' sub-folder contains images are highly possible to be meteors")
        print("The 'removed' sub-folders contain images filtered out")
        print("\nYou could further remove images from the 'good' folder which you don't think they")
        print("are really meteors")
        print("Or you could also move images from the 'removed' folders to the 'good' folder if")
        print("you think the images are actually meteors")
        print("\nAfter this is done, you can proceed to use this command to extract the meteors out of the background:")
        print("    "'auto_meteor_shower extraction {}'"".format(original_dir))

    if do_option == "all" or do_option == "extraction":
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
        my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)

        print("\nProcess finished!")
