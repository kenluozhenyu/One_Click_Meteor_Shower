import os
import sys

import detection
import gen_mask

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) < 2:
        print("\nUsage: auto_meteor_shower <option> <folder name>")
        print("option: all (Do for both detection and extraction)")
        print("        detection  (This is the step 1. Detection only)")
        print("        extraction (This is the step 2. Extraction only)")
        sys.exit(1)

    do_option = argv[0]
    if do_option != "all" and do_option != "detection" and do_option != "extraction":
        print("\nUsage: auto_meteor_shower <option> <folder name>")
        print("option: all (Do for both detection and extraction)")
        print("        detection  (This is the step 1. Detection only)")
        print("        extraction (This is the step 2. Extraction only)")
        sys.exit(1)

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
    extracted_dir = os.path.join(process_dir, '2_cropped')

    gray_256_dir = os.path.join(process_dir, '3_gray_256')
    mask_256_dir = os.path.join(process_dir, '4_mask_256')
    mask_resize_back_dir = os.path.join(process_dir, '5_mask_resize_back')
    # mask_extended_back_dir = os.path.join(process_dir, '6_mask_extended_back')
    object_extracted_dir = os.path.join(process_dir, '6_object_extracted')
    FINAL_dir = os.path.join(process_dir, '7_FINAL')
    FINAL_combined_dir = os.path.join(process_dir, '8_FINAL_combined')

    if do_option == "all" or do_option == "detection":
        meteor_detector.detect_n_extract_meteor_from_folder(original_dir, process_dir, verbose=1)

    if do_option == "detection":
        print("\n************************************************************")
        print("Possible objects detection finished.")
        print("You may go to the "'2_cropped'" folder to double check the detection objects.")
        print("And just delete those you don't think they are meteors.")
        print("\nAfter this is done, you can proceed to use this command to extract the meteors out of the background:")
        print("    "'auto_meteor_shower extraction {}'"".format(original_dir))

    if do_option == "all" or do_option == "extraction":
        my_gen_mask.convert_image_folder_to_gray_256(extracted_dir, gray_256_dir)
        my_gen_mask.gen_meteor_mask_from_folder(gray_256_dir, mask_256_dir)
        my_gen_mask.resize_mask_to_original_cropped_size(mask_256_dir, mask_resize_back_dir)

        my_gen_mask.extract_meteor_from_cropped_folder_with_mask(extracted_dir,
                                                                 mask_resize_back_dir,
                                                                 object_extracted_dir,
                                                                 verbose=1)
        my_gen_mask.extend_extracted_objects_to_original_photo_size(object_extracted_dir, FINAL_dir)
        my_gen_mask.combine_meteor_images_to_one(FINAL_dir, FINAL_combined_dir, verbose=1)
        print("\nProcess finished!")
