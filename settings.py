# -*- coding: utf-8 -*-

DETECTION_BLUR_KERNEL_SIZE_FOR_EQUATORIAL_MOUNTED_IMAGES = 3
DETECTION_BLUR_KERNEL_SIZE_FOR_FIXED_TRIPOD_IMAGES = 5
# DETECTION_BLUR_KERNEL_SIZE_FOR_DIRECT_METHOD = 11
DETECTION_CANNY_LOW_THRESHOLD = 50
DETECTION_CANNY_RATIO = 3
DETECTION_CANNY_KERNEL_SIZE = 3
DETECTION_LINE_THRESHOLD = 20
DETECTION_LINE_MIN_LINE_LENGTH = 30
DETECTION_LINE_MAX_LINE_GAP = 3

# DETECTION_CROP_IMAGE_BOX_SIZE = 640
DETECTION_CROP_IMAGE_BOX_SIZE = 256

# If cropped image size > DETECTION_CROP_IMAGE_BOX_SIZE * ratio
# will be divided to mosaic images, so as to improve the resolution
RATIO_FOR_MOSAIC = 1.5

# When dividing a big image to mosaic, each image would need
# to have some overlap.
# This ratio is about how much (of the width or height) of
# two images to be overlapped
# Value from 0 to 1
# MOSAIC_OVERLAP_RATIO = 0.5
MOSAIC_OVERLAP_RATIO = 0.25

# To ensure the detected object is within the cropped image, would
# need to enlarge the detection box
# DETECTION_CROP_IMAGE_BOX_FACTOR = 1.5
DETECTION_CROP_IMAGE_BOX_FACTOR = 3.0

# When merging two detection box, need to have some level of overlap
# This threshold means the centers in both dimension need to be less
# than (box1_width + box2_width) * threshold
# BOX_OVERLAP_THRESHOLD = 0.5
BOX_OVERLAP_THRESHOLD = 0.2

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
# For meteor object extraction
#
# To avoid the dark boundary of the extracted meteor object is too thick.
# If the specific pixel has RGB value lower than this threshold, it will
# be replaced with transparent background.
#
# This value would need to be adjusted according to the brightness of the
# background image
#
# - If the extracted meteor object got cut in the head/tail too much, try to
#   lower this value (as long as the mask file covers the meteor body).
# - If the extracted meteor object has the edge too thick/too dark compare
#   to the star background image, try to increase this value (better to be
#   within 80).
EXTRACT_RGB_VALUE_THRESHOLD = 48

# =============================================================================
# For multi-thread processing
# To avoid memory exhaustion sometimes we need to control the maximum core #
MAX_CPU_FOR_DETECTION = 24
MAX_CPU_FOR_MASK_EXTRACTION = 24

# =============================================================================
# The Neural Network
CNN_IMAGE_SIZE = 256
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201008_2.375-0.054.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201008_3.187-0.040.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201008_4.378-0.162.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201017_4_cnn4_.459-0.151.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201017_4_cnn4_lre-4_.814-0.097.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201020_1_cnn7_lre-5_.259-0.114.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201022_1_cnn7_lre-4_.483-0.00004.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201025_1_cnn11_lre-4_.732-0.00001.hdf5'

# 2020-10-25: This model weight seems to have the best performance on one test set
CNN_SAVED_MODEL = 'saved_model/cnn_star_256_20201025_1_cnn11_lre-4_.731-0.00002.hdf5'

# CNN_SAVED_MODEL = 'saved_model/cnn_star_256-20200517-0.074.hdf5'
# CNN_SAVED_MODEL = 'saved_model/cnn_star_256-20200518-0.091.hdf5'

UNET_IMAGE_SIZE = 256
# unet_saved_model = 'saved_model/unet_meteor_gray_256_png.hdf5'
# UNET_SAVED_MODEL = 'saved_model/unet_meteor_gray_256_png-good.hdf5'

# 2020-11-17: Test result on a Qing-Hai dataset shows this model
# seems still better than the result from the UNET++ model
# Need to re-do the test with mosaic ratio back to 0.5 <== TODO
# UNET_SAVED_MODEL = 'saved_model/unet_meteor_gray256png_final.hdf5'

# 2020-11-11: Test result shows the 20201108-2 version is a little bit better than
#             the 20201110-1 version
# UNET_SAVED_MODEL = 'saved_model/unet++_meteor_gray256png_20201108-2_wo_val.183-0.228.hdf5'
# UNET_SAVED_MODEL = 'saved_model/unet++_meteor_gray256png_20201110-1_wo_val.183-0.220.hdf5'

# 2021-03-15: Newer trained model for UNET++. Test result is quite good
UNET_SAVED_MODEL = 'saved_model/unet++_meteor_gray256_20210314-3_wo_val.297-0.201.hdf5'