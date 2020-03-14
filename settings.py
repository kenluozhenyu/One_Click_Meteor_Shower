# -*- coding: utf-8 -*-

DETECTION_BLUR_KERNEL_SIZE = 3
DETECTION_CANNY_LOW_THRESHOLD = 50
DETECTION_CANNY_RATIO = 3
DETECTION_CANNY_KERNEL_SIZE = 3
DETECTION_LINE_THRESHOLD = 50
DETECTION_LINE_MIN_LINE_LENGTH = 30
DETECTION_LINE_MAX_LINE_GAP = 3

DETECTION_CROP_IMAGE_BOX_SIZE = 640

# To ensure the detected object is within the cropped image, would
# need to enlarge the detection box
# DETECTION_CROP_IMAGE_BOX_FACTOR = 1.5
DETECTION_CROP_IMAGE_BOX_FACTOR = 3.0

# When merging two detection box, need to have some level of overlap
# This threshold means the centers in both dimension need to be less
# than (box1_width + box2_width) * threshold
BOX_OVERLAP_THRESHOLD = 0.5

# =============================================================================
# For checking false detection
#
# If the detected lines are very closed, and be parallel
# with the border, ignore them
# Value is # of pixels
DETECTION_IMAGE_BORDER_THRESHOLD = 8
LINE_X_OR_Y_DELTA_THRESHOLD = 6

# Get some points around the center of the detected line
# Get those points' color, to check if RGB(0,0,0) which
# means it is on the edge
# Value is # of pixels
LINE_CENTER_RADIUS_CHECKING = 10

# If the pixel color is less than this value, consider it is from the border
DETECTION_BORDER_COLOR_THRESHOLD = 9

# =============================================================================
# For merging two short lines which should belong to the same one
#
# 1 )Acceptable angle delta (in rad, 0.01 rad is around 0.6 degree)
#    Value is angel in rad
#    ~ 23 deg
LINE_ANGEL_DELTA_THRESHOLD = 0.4

# 2) If the two lines are closed enough
# Value is # of pixels
# Would need to consider changing this to ration(% of img size)
LINE_VERTICAL_DISTANCE_FOR_MERGE_THRESHOLD = 8
LINE_VERTICAL_DISTANCE_FOR_MERGE_W_OVERLAP_THRESHOLD = 3
# ~0.06 of the img width
LINE_DISTANCE_FOR_MERGE_THRESHOLD = 300

# May not use this one
LINE_JOINT_ANGEL_DELTA_THRESHOLD = 0.4

# =============================================================================
# For satellite checking
#
# If two lines (have similar angel) are closed enough
# to be considered as satellite
# Value is # of pixels
# Would need to consider changing this to ration(% of img size)
# And would need to be calculated from
#   1) Lens' focal length (different angle width)
#   2) Image interval (# of sec)
LINE_VERTICAL_DISTANCE_FOR_SATELLITE_THRESHOLD = 12
LINE_DISTANCE_FOR_SATELLITE_THRESHOLD = 300

# =============================================================================
# The Neural Network
UNET_IMAGE_SIZE = 256
# unet_saved_model = 'saved_model/unet_meteor_gray_256_png.hdf5'
UNET_SAVED_MODEL = 'saved_model/unet_meteor_gray_256_png-good.hdf5'
