import os
import sys

import detection
# import gen_mask

if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) == 0:
        print("Usage: two_clicks_step1.py <folder name>")
        sys.exit(1)

    original_dir = argv[0]
    if not os.path.exists(original_dir):
        print("No such directory: {}".format(original_dir))
        sys.exit(1)

    meteor_detector = detection.MeteorDetector()
    # my_gen_mask = gen_mask.Gen_mask()

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '2_cropped' is hardcoded, don't change
    # Other sub-folder names in below can be changed
    extracted_dir = os.path.join(process_dir, '2_cropped')

    meteor_detector.detect_n_extract_meteor_from_folder(original_dir, process_dir, verbose=1)

    print("\nPossible objects extraction finished.")
    print("You may go to the "'2_cropped'" folder to double check the detection objects.")
    print("And just delete those you don't think they are meteors.")
    print("\nAfter this is done, you can proceed to use this command to extract the meteors out of the background:")
    print("    "'two_clicks_step2.py {}'"".format(original_dir))
