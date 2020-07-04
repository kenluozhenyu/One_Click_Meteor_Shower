import os
import sys

import detection
# import gen_mask

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("\nUsage: two_clicks_step1 <equatorial_mount option: Y/N> <folder name>")
        print("equatorial_mount option: Y (If the images were taken on equatorial mount)")
        print("                         N (Choose this for images taken on fixed tripod)")
        sys.exit(1)

    equatorial_mount_option = argv[0]
    equatorial_mount_option = equatorial_mount_option.upper()
    if equatorial_mount_option != "Y" and equatorial_mount_option != "N":
        print("\nUsage: two_clicks_step1 <equatorial_mount option: Y/N> <folder name>")
        print("equatorial_mount option: Y (If the images were taken on equatorial mount)")
        print("                         N (Choose this for images taken on fixed tripod)")
        sys.exit(1)

    original_dir = argv[1]
    if not os.path.exists(original_dir):
        print("No such directory: {}".format(original_dir))
        sys.exit(1)

    # meteor_detector = detection.MeteorDetector()

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

    if equatorial_mount_option == 'Y':
        detection.multi_thread_process_detect_n_extract_meteor_from_folder(original_dir,
                                                                           process_dir,
                                                                           subtraction=True,
                                                                           equatorial_mount=True,
                                                                           verbose=1)
    else:
        detection.multi_thread_process_detect_n_extract_meteor_from_folder(original_dir,
                                                                           process_dir,
                                                                           subtraction=True,
                                                                           equatorial_mount=False,
                                                                           verbose=1)

    # meteor_detector.filter_possible_not_meteor_objects(extracted_dir, keep_dir, not_sure_dir, removed_dir)
    # meteor_detector.filter_possible_not_meteor_objects(extracted_dir, keep_dir, removed_dir)

    detection.filter_possible_not_meteor_objects(extracted_dir, keep_dir, removed_dir)

    print("\n======================================================================================================")
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
    print("    "'two_clicks_step2 {}'"".format(original_dir))
