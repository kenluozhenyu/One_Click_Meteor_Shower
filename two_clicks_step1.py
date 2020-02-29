import os
import sys

import detection
# import gen_mask

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
    # my_gen_mask = gen_mask.Gen_mask()

    # Below sub-folders will be created by the program
    process_dir = os.path.join(original_dir, 'process')

    # Need to have below sub-folders
    # The '2_extraction' is hardcoded, don't change
    # Other sub-folder names in below can be changed
    extracted_dir = os.path.join(process_dir, '2_extraction')

    meteor_detector.detect_n_extract_meteor_from_folder(original_dir, process_dir, verbose=1)
