
DETECTION_BLUR_KERNEL_SIZE = 3
DETECTION_CANNY_LOW_THRESHOLD = 40
DETECTION_CANNY_RATIO = 3
DETECTION_CANNY_KERNEL_SIZE = 3
DETECTION_LINE_THRESHOLD = 50
DETECTION_LINE_MIN_LINE_LENGTH = 30
DETECTION_LINE_MAX_LINE_GAP = 3

# If the detected lines are very closed, and be parallel
# with the border, ignore them
DETECTION_IMAGE_BORDER_THRESHOLD = 10
DETECTION_LINE_X_OR_Y_DELTA_THRESHOLD = 3

DETECTION_CROP_IMAGE_BOX_SIZE = 640
DETECTION_CROP_IMAGE_BOX_FACTOR = 1.5

# If the detected lines are very closed, and be parallel
# with the border, ignore them
DETECTION_IMAGE_BORDER_THRESHOLD = 8
LINE_X_OR_Y_DELTA_THRESHOLD = 6

# Get some points around the center of the detected line
# Get those points' color, to check if RGB(0,0,0) which
# means it is on the edge
LINE_CENTER_RADIUS_CHECKING = 20

# For merging two short lines which should belong to the same one
# Acceptable angle delta (in rad, around 0.6 degree)
LINE_ANGEL_DELTA_THRESHOLD = 0.01

# If two line (have the same angel) are closed enough
LINE_DISTANCE_FOR_MERGE_THRESHOLD = 100

# If the pixel color is less than this value, consider it is from the border
DETECTION_BORDER_COLOR_THRESHOLD = 3

UNET_IMAGE_SIZE = 256
# unet_saved_model = 'saved_model/unet_meteor_gray_256_png.hdf5'
UNET_SAVED_MODEL = 'saved_model/unet_meteor_gray_256_png-good.hdf5'


