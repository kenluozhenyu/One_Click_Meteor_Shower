
DETECTION_BLUR_KERNEL_SIZE = 3
DETECTION_CANNY_LOW_THRESHOLD = 40
DETECTION_CANNY_RATIO = 3
DETECTION_CANNY_KERNEL_SIZE = 3
DETECTION_LINE_THRESHOLD = 50
DETECTION_LINE_MIN_LINE_LENGTH = 30
DETECTION_LINE_MAX_LINE_GAP = 3

# If the detected lines are very closed, and be parallel
# with the border, ignore them
# Value is # of pixels
DETECTION_IMAGE_BORDER_THRESHOLD = 10
DETECTION_LINE_X_OR_Y_DELTA_THRESHOLD = 3

DETECTION_CROP_IMAGE_BOX_SIZE = 640
DETECTION_CROP_IMAGE_BOX_FACTOR = 1.5

# If the detected lines are very closed, and be parallel
# with the border, ignore them
# Value is # of pixels
DETECTION_IMAGE_BORDER_THRESHOLD = 8
LINE_X_OR_Y_DELTA_THRESHOLD = 6

# Get some points around the center of the detected line
# Get those points' color, to check if RGB(0,0,0) which
# means it is on the edge
# Value is # of pixels
LINE_CENTER_RADIUS_CHECKING = 20

# For merging two short lines which should belong to the same one
# Acceptable angle delta (in rad, 0.01 rad is around 0.6 degree)
# Value is angel in rad
LINE_ANGEL_DELTA_THRESHOLD = 0.4
LINE_JOINT_ANGEL_DELTA_THRESHOLD = 0.4

# If two lines (have the same angel) are closed enough
# to be consider as one line and to be merged
# Value is # of pixels
# Would need to consider changing this to ration(% of img size)
LINE_DISTANCE_FOR_MERGE_THRESHOLD = 100

# If two lines (have similar angel) are closed enough
# to be considered as satellite
# Value is # of pixels
# Would need to consider changing this to ration(% of img size)
LINE_DISTANCE_FOR_SATELLITE_THRESHOLD = 200

# If the pixel color is less than this value, consider it is from the border
DETECTION_BORDER_COLOR_THRESHOLD = 3

# When merging two detection box, need to have some level of overlap
# This threshold means the centers in both dimension need to be less
# than (box1_width + box2_width) * threshold
BOX_OVERLAP_THRESHOLD = 0.5

UNET_IMAGE_SIZE = 256
# unet_saved_model = 'saved_model/unet_meteor_gray_256_png.hdf5'
UNET_SAVED_MODEL = 'saved_model/unet_meteor_gray_256_png-good.hdf5'
