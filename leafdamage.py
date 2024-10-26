"""
This module computes leaf damage ratio index from UAV RGB plot 
"""

import cv2
import numpy as np
from uavhelpers import safefloat, natural_round
import types


def process(tile_obj, **kwargs) -> dict:
    """
    This function calculate leaf damage index based on a plot RGB image

    Args
        tile_obj:

    Returns
        a json object with the leaf damage index values
    """
    if 'keep_artifact_ep' in kwargs:
        keep_artifact = kwargs['keep_artifact_ep']
    else:
        keep_artifact = None
    call_ka = isinstance(keep_artifact, types.FunctionType)

    ldi_reason = ""
    result = {"ldi": None, "ldi_reason": ldi_reason}

    # tile_properties = deepcopy(tile_obj.tile_properties)

    plot_image = tile_obj.get_rotate_by_heading_output(reprocess=False)[0]

    if 'ldi_thresh' in tile_obj.config_model_parameters:
        ldi_thresh = tile_obj.config_model_parameters['ldi_thresh']
    else:
        ldi_thresh = dict(inset=0.1, value=40, brown_low_thresh=15, brown_sat_low_thresh=0)

    image_shape = plot_image.shape
    if len(image_shape) < 3 or image_shape[2] < 4:
        print("ERROR - Load of input image for LDI analysis must be three band RGB.")

    # the original version of this code called cv2.imread instead of GDAL
    # however, cv2.imread will not open >4 band inputs, so GDAL is being used for greater generalization
    #
    # emulate OpenCV RGB order (cv2.imread uses BGR when loading)
    plot_image = plot_image[..., [2, 1, 0]]  # make this BGR while also dropping bands in excess of the first three

    # NOTE:  Multi-spect input is not full RGB, an attribute which can substantially change what that LDI values
    # would represent.
    # NOTE:  thresholds coded herein are generally selected for best operation with Sony RX1 RGB in conjunction
    # with its standard configuration of exposure settings.  Other inputs may not generate consistent LDI values.

    image_shape = plot_image.shape
    plot_image = plot_image.astype('float')
    image_max = plot_image.max()
    if image_max <= 1:
        plot_image *= 255
    if image_max > 255:
        # This input data type had a high bit depth than the usual RGB camera output uses.
        # much more likely when dealing with non-true RGB camera sources (multi-spectal)
        plot_image /= image_max
        plot_image *= 255
    plot_image = plot_image.astype('uint8')

    # TODO? - use new band descriptions to map RGB/rgb sources; however, this implicit operation should
    #         be consistently correct since either rgb source is acceptable and RGB is given first
    #         for eight band inputs (i.e. Sentera 6x with both forms of RGB)

    # TODO - consider adding row level metrics (insets would become a masking regions instead)

    # default border_ratio value is a 10% per side inset (20% of entire width/height)
    # ASSERTION:  input data has not already been inset
    delta_1 = int(natural_round(image_shape[0] * ldi_thresh['inset']))
    delta_2 = int(natural_round(image_shape[1] * ldi_thresh['inset']))

    plot_img_trimmed = plot_image[delta_1:image_shape[0] - delta_1, delta_2:image_shape[1] - delta_2, :]
    image_shape = plot_image.shape

    # Constrain the background (Important, Don't remove!!)
    hsv_image = cv2.cvtColor(plot_img_trimmed, cv2.COLOR_BGR2HSV)
    plot_img_trimmed[hsv_image[:, :, 2] < ldi_thresh['value'], :] = [0, 0, 0]

    # Enhancement
    stretched_img = apply_CLAHE(plot_img_trimmed)

    # Calculate the LDI based on color scheme
    hsv_image_2 = cv2.cvtColor(stretched_img, cv2.COLOR_BGR2HSV)

    hue_layer = hsv_image_2[:, :, 0]
    hue_layer = hue_layer[hue_layer > 0]

    hue_data_array = np.ndarray.flatten(hue_layer)
    mean_color_h = hue_data_array.mean()

    #
    # sat_layer = hsv_image_2[:, :, 1]
    # sat_layer = sat_layer[sat_layer > 0]
    # std_color_h = hue_data_array.std()
    # sat_data_array = np.ndarray.flatten(sat_layer)
    # mean_color_sat = sat_data_array.mean()
    # std_color_sat = sat_data_array.std()

    # A decision tree for dynamic thresholding based on the hue value
    brown_low_thresh = ldi_thresh['brown_low_thresh']
    brown_sat_low_thresh = ldi_thresh['brown_sat_low_thresh']

    if mean_color_h > 50:
        threshold_h = 50  #
        brown_upper_thresh = 38
    elif mean_color_h > 40:
        threshold_h = 41
        brown_upper_thresh = 35
    elif mean_color_h > 30:  # Fuzzy area [60-80 degrees]
        threshold_h = int(mean_color_h)
        # brown_upper_thresh = 30
        if mean_color_h > 36:
            brown_upper_thresh = 35
        else:
            brown_upper_thresh = int(mean_color_h) - 2
    else:
        # Add an additional filter on saturation to constrain the soil background
        threshold_h = 30
        brown_sat_low_thresh = 30
        if mean_color_h > 25:  # [50-60 degrees]
            brown_upper_thresh = int(mean_color_h)
        else:
            brown_low_thresh = 16
            brown_upper_thresh = 25

    # Normal masking algorithm
    lower_green = np.array([threshold_h, 0, 0])
    upper_green = np.array([75, 255, 255])

    lower_brown = np.array([brown_low_thresh, brown_sat_low_thresh, 0])
    upper_brown = np.array([brown_upper_thresh, 255, 255])

    mask_green = cv2.inRange(hsv_image_2, lower_green, upper_green) / 255
    mask_brown = cv2.inRange(hsv_image_2, lower_brown, upper_brown) / 255

    # total_pixel_number = mask_green.shape[0] * mask_green.shape[1]

    # Generating features
    # green_ratio = float(np.sum(mask_green[:, :])) / float(total_pixel_number)
    # brown_ratio = float(np.sum(mask_brown[:, :])) / float(total_pixel_number)

    # Generation LDI value based on brown and green ratio values
    brown_green_ratio = float(np.sum(cv2.bitwise_or(mask_green, mask_brown)))
    if brown_green_ratio == 0:
        brown_green_ratio = None
        ldi_reason = "No brown or green raster pixels were found; LDI measurement omitted."
    else:
        brown_green_ratio = round(float(np.sum(mask_brown[:, :])) / brown_green_ratio, 3)

    # # return color feature vector and LDI value, for debugging purpose only
    # result.update({'h_mean': round(mean_color_h / 180, 3), 'h_std': round(std_color_h / 180, 3),
    #                's_mean': round(mean_color_sat / 360, 3), 's_std': round(std_color_sat / 360, 3),
    #                'green_ratio': round(green_ratio, 3), 'brown_ratio': round(brown_ratio, 3),
    #                'ldi': brown_green_ratio})
    result.update({"ldi": safefloat(brown_green_ratio), "ldi_reason": ldi_reason})
    return result


def apply_CLAHE(bgr_image):
    """
    Enhance color image to mitigate lighting variation
    :param bgr_image: three bands color image
    :return: corrected color image
    """
    plotImg_trimmed_lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(40, 40))
    lab_tiles = cv2.split(plotImg_trimmed_lab)
    lab_tiles = list(lab_tiles)
    lab_tiles[0] = clahe.apply(lab_tiles[0])
    lab_image = cv2.merge(lab_tiles)
    stretched_img = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return stretched_img


def rating_from_LDI(LDI_val):
    """
    Convert the LDI score into disease rating in [1, 9]
    :param LDI_val: integer or vector
    :return: integer or vector
    """
    # This mapping function can be calibrated with extended validation dataset
    rating = 9 * LDI_val + 0.5
    # 0.5>=rating<=9.5
    # for np.round() operations, 0.5 and 9.5 would round to 0 and 10 respectively, so use a substitute rounder
    # and constrain the rating to be 1-9 only
    rating = max(1, min(9, natural_round(rating)))
    return int(rating)
