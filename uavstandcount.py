# -*- coding: future_fstrings -*-
import cv2
from copy import deepcopy, copy
import json
import math
import numpy as np
import os
import peakutils
import rasterio
import regex as re
import scipy.signal as signal
import types
from UAVAnalyticObject import UAVAnalyticObject, get_analytic_object
from uavhelpers import get_planting_terms, add_row_labels, rotate_plot, gdal_open, gdal_update, gdal_close,\
    denoise_mask, natural_round, safefloat
from uavdeepstand import sagemaker_preprocess as uavdeepstand_sagemaker_preprocess

DEBUG = False


def sagemaker_preprocess(obj_path, metadata, **kwargs):
    this_metadata = deepcopy(metadata)
    tile_obj = None
    if isinstance(this_metadata, dict):
        if 'analytic_object' in this_metadata:
            tile_obj = this_metadata['analytic_object']['tile']
        elif isinstance(obj_path, str):
            # legacy call; make an object for backward compatibility
            tile_obj = get_analytic_object(obj_path, this_metadata)
            this_metadata['analytic_object']['tile'] = tile_obj

    if tile_obj is None:
        raise ValueError("Analytic object creation failure.")

    return uavdeepstand_sagemaker_preprocess(obj_path, this_metadata, **kwargs)


def process(tile_obj: UAVAnalyticObject = None, plant_labeling=False, plant_loc_csv=False,
            method=None, **kwargs) -> dict:
    """
    A function to return a json object from stand computation
    Args
        tile_path: string
            input : raw mosaic images generated from tile clipping
        metadata: dictionary object
            metadata: metadata information for a single plot
    Returns
        a dictionary object containing all stand metric fields
    """

    # General expectations for getting a valid stand count result:
    # 1. crop = corn
    # 2. growth_stage in [v2, v3, v4]; defaults to v3
    # 3. target_seed_count or sdplt are not null and > 0.
    # 4. valid required meta is provided for:
    #    plot_id, rows_per_plot, harvest_rows_per_plot, planted_length_inches
    #    target_seed_count and row_spacing_inches
    #
    # Items 1 and 2 are not strictly enforced within this code or related subroutines.
    # Instead, where relevant, defaults are imposed as needed (e.g. v3 and/or corn assumed)
    # Item 3's sdplt is used in legacy (disabled) uavmetricqaqc only, thus stand can be calculated without it.

    results = {'stand': None, 'row_stand': None, 'row_stand_xin_byplant': None, 'row_stand_yin_byplant': None,
               'observed_rows': None, 'counted_rows': None, 'stand_source': None, 'stand_reason': ''}

    if len(tile_obj.get_config_status):
        # this is here to support inclusion of stand count in recipes to which it may not actually apply for
        # every data set (e.g. Field Solution examples without target seed count)
        #
        # the config.json 'config_parameter_failure_mode'='SOFT' setting is what routes config exceptions here

        tile_obj.mprint(f"Attempted configuration failed, reporting: '{tile_obj.get_config_status}'")
        tile_obj.mprint_dump()

        results.update({"stand_reason": tile_obj.get_config_status})
        return results

    metadata = deepcopy(tile_obj.tile_meta)

    plot_properties = metadata["properties"]

    meta_plot_row = plot_properties["rows_per_plot"]
    meta_harv_row = plot_properties["harvest_rows_per_plot"]
    if meta_plot_row < 3:
        meta_harv_row = meta_plot_row
        plot_properties["harvest_rows_per_plot"] = meta_plot_row

    call_results = results
    if meta_harv_row > meta_plot_row:
        msg = f"Harvest row count of {meta_harv_row} for {meta_plot_row} row plot is not valid; excluded."
        tile_obj.mprint(f"ERROR - {msg}")
        results.update({'stand_reason': msg})
    elif meta_harv_row % 2 != meta_plot_row % 2:
        # both terms must be even or odd, never mixed
        msg = f"Ambiguous harvest row positions for count of {meta_harv_row} in {meta_plot_row} row plot; excluded."
        tile_obj.mprint(f"ERROR - {msg}")
        results.update({'stand_reason': msg})
    else:
        peak_algorithm = tile_obj.config_model_parameters['metric_source'].upper().strip()
        if "stand_peak_algorithm" in tile_obj.model_parameters:
            user_peak_algorithm = tile_obj.model_parameters["stand_peak_algorithm"].strip()
            palist = tile_obj.config_model_parameters['peak_stand_algorithms']
            if not isinstance(user_peak_algorithm, str) or len(user_peak_algorithm) == 0:
                user_peak_algorithm = peak_algorithm
            if user_peak_algorithm in palist:
                peak_algorithm = user_peak_algorithm.upper()
            else:
                tile_obj.mprint(f"WARNING - Metric extent parameter value '{user_peak_algorithm}' not one of {palist};"
                                f" ignored, using '{peak_algorithm.lower()}'.")

        call_results = stand_count(tile_obj, results, plant_labeling, plant_loc_csv, method=method, **kwargs)[0]

        # Copy the configured source stand value into the primary stand outputs, making it possible to control
        # this assignment externally.
        peak_algorithm = peak_algorithm.lower()
        for key_out in tile_obj.config_model_parameters['metric_assignments']:
            key_in = tile_obj.config_model_parameters['metric_assignments'][key_out]
            key_in = re.sub("%", peak_algorithm, key_in)
            if key_in in call_results:
                call_results.update({key_out: call_results[key_in]})
            else:
                tile_obj.mprint(f"ERROR - {key_in} not found; {peak_algorithm} data not assigned as output.")
        call_results.update({'stand_source': peak_algorithm})

    tile_obj.mprint_dump()

    results.update(call_results)
    return results


def check_is_harvest_row(rows_per_plot, harvest_rows_per_plot, row_idx):
    is_harvest_row = True

    row_diff = (rows_per_plot - harvest_rows_per_plot)
    row_buffer = math.floor(row_diff / 2)

    if row_idx < row_buffer or row_idx >= (rows_per_plot - row_buffer):
        is_harvest_row = False

    return is_harvest_row


def stand_count(tile_obj,
                stand_results: dict = None,
                plant_labeling: bool = False,
                plant_loc_csv: bool = True,
                metric_extent: str = "PLOT",
                **kwargs) -> tuple:
    """
    There are three cases for corn stand count per plot: 1) two-row plot harvest two rows; 2) four-row plot harvest
    center two rows; 3) four-row plot harvest all four rows.
    :param tile_obj:
    :param stand_results:
    :param plant_labeling: output plant-labeled image
    :param plant_loc_csv: plant location csv
    :param metric_extent: the string selection controlling whether metrics are
                          constrained to just the planing extent or not
    :return: corn stand count in JSON format, plot location table
    """
    if 'keep_artifact_ep' in kwargs:
        keep_artifact = kwargs['keep_artifact_ep']
    else:
        keep_artifact = None
    call_ka = isinstance(keep_artifact, types.FunctionType)

    if DEBUG:
        tile_obj.mprint("DEBUG - Processing Started")

    # metadata = deepcopy(tile_obj.tile_meta)  # localize scope
    plot_properties = deepcopy(tile_obj.tile_properties)

    # Required meta fields; failed assignments will case an abend here.
    # The pipeline response to an abend is the halt of all related recipe activity for a given mosaic.
    plot_id = plot_properties["plot_id"]
    meta_plot_row = plot_properties["rows_per_plot"]
    meta_harv_row = plot_properties["harvest_rows_per_plot"]
    meta_row_spacing = plot_properties["row_spacing_inches"]
    # target_seed_count and meta_planted_length will also be required by the get_planting_terms call

    model_parameters = deepcopy(tile_obj.model_parameters)

    metric_extent_specified = False
    if "metric_extent" in model_parameters:
        mevalue = model_parameters["metric_extent"].upper().strip()
        melist = ["PLOT", "PLANTING"]
        if mevalue in melist:
            metric_extent = mevalue
            metric_extent_specified = True
        else:
            tile_obj.mprint(f"WARNING - Metric extent parameter value '{mevalue}' not one of {melist};"
                            f" ignored, using '{metric_extent}'.")

    # TODO? - resize all input masks to a uniform 0.1 cm pixel size

    is_harv_row = []
    for r in range(meta_plot_row):
        is_harv_row.append(check_is_harvest_row(meta_plot_row, meta_harv_row, r))

    rgb_bands, _, alpha, heading, rinfo = tile_obj.get_rotate_by_heading_output()
    mask_img = tile_obj.metric_mask.copy()

    yx_m = rinfo['yx_m']
    yx_cm = yx_m * 100
    ppi_l = 2.54 / yx_cm[1]  # x-axis mapped to length here for rotation applied standardization
    ppi_w = 2.54 / yx_cm[0]  # y-axis mapped to width here for rotation applied standardization

    plant_loc_table = ['algorithm,plot_id,ccw_rot,x_cm,y_cm,row_id,is_harvest_row,LocX,LocY,algorithm_data']

    # iADN start
    if DEBUG:
        tile_obj.mprint(f"DEBUG - PPI(along row)={ppi_l}px, PPI(across row)={ppi_w}")

    # target_alt = int(float(metadata["flight"]["target_alt"]))
    # if target_alt != 40 and target_alt != 60:
    #     tile_obj.mprint(f"WARNING - Unexpected target altitude={target_alt}m")
    # iADN end

    # inter-plant spacing calculation added on 02/11/2020 by iADN
    meta_planted_length, ips, planting_pop = \
        get_planting_terms(plot_properties, planted_length_default=None, ips_required=True)

    min_peak_interval = math.ceil(ips * ppi_l / 2.0)  # use half IPS separation between plants expected at the minimum

    tile_obj.mprint(f"INFO - IPS={ips:0.3f}\"; PPI={ppi_l:0.3f}px; min_peak_interval={min_peak_interval}px;"
                    f" population={planting_pop}/acre")

    row_extents = None
    stand_row = [None] * meta_plot_row
    mask_is_empty = mask_img.max() == 0
    stand_reason = ""
    peak_stand_reason = ""
    object_stand_reason = ""
    full_row_sei = [0, mask_img.shape[1] - 1]
    num_row = 0

    num_row_qc_obs, row_idx, rot_mask_img = tile_obj.get_detect_rows_output()
    num_row_original = len(row_idx)

    stand_results.update({"peak_stand": None, "row_peak_stand": None,
                          "row_peak_stand_xin_byplant": None,
                          "row_peak_stand_yin_byplant": None,
                          "peak_stand_reason": None})

    stand_results.update({"object_stand": None, "row_object_stand": None,
                          "row_object_stand_xin_byplant": None,
                          "row_object_stand_yin_byplant": None,
                          "object_stand_reason": None})

    peak_stand_done = False
    if mask_is_empty:
        stand_reason = "No canopy detected."
        tile_obj.mprint(f"WARNING - {stand_reason}")
    elif min_peak_interval is None:
        stand_reason = "min_peak_interval computed as 'None'."
        tile_obj.mprint(f"ERROR - {stand_reason}")
    elif num_row_original == 0:
        stand_reason = "No rows observed within plot."
        tile_obj.mprint(f"WARNING - {stand_reason}")
    else:
        peak_stand_done = True

        # 31 was hard coded (30+1 with 30 representing 6" worth of 0.5cm pixels)
        plant_window_px = math.ceil(ips * ppi_l)
        if plant_window_px % 2 == 0:
            # window lengths must be odd
            plant_window_px += 1

        tile_obj.mprint(f"INFO - Peak stand plant_window_px={plant_window_px}px;"
                        f" {plant_window_px / ppi_l:0.03f}\" @ {ppi_l}ppi")

        row_masks, row_extents, planted_row_sei = tile_obj.get_crop_rows_output()

        if metric_extent_specified and metric_extent == "PLOT":
            object_row_sei = copy(full_row_sei)
        else:
            # default for object is planting row!
            object_row_sei = copy(planted_row_sei)

        # TODO - consider a mode of metric output where PLANTING is additive to PLOT and can run
        #        side-by-side without publication for a while
        if metric_extent == "PLANTING":
            peak_row_sei = copy(planted_row_sei)
        else:
            # default for peak is full row!
            peak_row_sei = copy(full_row_sei)

        # For QC purposes, don't count rows which are less than 90% the width of the widest row found:
        extent_widths = []
        row_key_list = []
        for row_key in sorted(row_extents.keys()):
            row_key_list.append(row_key)
            extent_widths.append(row_extents[row_key][1] - row_extents[row_key][0])

        extent_widths = np.abs(np.array(extent_widths))
        num_row_qc_obs -= max(0, len(extent_widths) - len(np.where(extent_widths > (extent_widths.max() * 0.9))[0]))

        if meta_plot_row != num_row_qc_obs:
            tile_obj.mprint(f"WARNING - The number of QC rows observed={num_row_qc_obs}; expected {meta_plot_row}")

        # fig, axs = plt.subplots(num_row_original, 1, figsize=(15, 6), facecolor='w', edgecolor='k')
        # for i in range(num_row_original):
        #     axs[i].imshow(row_masks[f"row_{i + 1}_mask"], cmap='gray')
        # plt.show()

        # older code had 20 hardcoded; that is an approximation for 4" @ 0.5cm per pixel
        # then based on typical 30" rows, that is ~13% of row width
        boundary_pct_row = tile_obj.config_model_parameters['boundary_pct_row']
        boundary_buffer = int(max(1, round(ppi_l * (meta_row_spacing * boundary_pct_row))))

        smoothing_polyorder = tile_obj.config_model_parameters['smoothing_polyorder']
        peakutils_thresh = tile_obj.config_model_parameters['peakutils_thresh']
        open_extent_cm = tile_obj.config_model_parameters['open_extent_cm']
        object_scale_bias_ips_distance_factor = tile_obj.config_model_parameters['object_scale_bias_ips_distance_factor']
        object_scale_bias_peak_drop_factor = tile_obj.config_model_parameters["object_scale_bias_peak_drop_factor"]
        same_object_peak_thresh_factor = tile_obj.config_model_parameters['same_object_peak_thresh_percent'] / 100
        open_extent_px = open_extent_cm / yx_cm[1]
        area_thresh_sqcm = tile_obj.config_model_parameters['area_thresh_sqcm']
        area_thresh_px = area_thresh_sqcm / np.prod(yx_cm)
        row_constrained_list = []
        overplanting_factor = 1 + (tile_obj.config_model_parameters['overplanting_percent'] / 100)
        row_stand_xin_byplant = [[None]] * num_row_original
        row_stand_yin_byplant = [[None]] * num_row_original

        for peak_metric_id in ["peak", "object"]:
            num_row = 0
            num_plant = 0

            for r in range(num_row_original):
                if row_idx[r] < boundary_buffer or row_idx[r] > rot_mask_img.shape[0] - 1 - boundary_buffer:
                    # summing of the related axis means this buffer need not be spatially derived
                    tile_obj.mprint(f"WARNING - Row {r + 1} is within {boundary_pct_row * 100}% row widths of the plot"
                                    f" edge; row {r + 1} stand not computed.")
                else:
                    # construct the col-side pixel count curve for finding the split point
                    idx_str = f"00{r + 1}"[-3:]

                    if peak_metric_id == "peak":
                        row_sei = peak_row_sei
                    else:
                        row_sei = object_row_sei

                    row_mask = row_masks[f'row_{idx_str}_mask'][:, row_sei[0]:row_sei[1] + 1]

                    rm_len = len(row_mask)
                    if rm_len % 2 == 0:
                        rm_len -= 1  # if this is -1 now, an abend is forthcoming
                    row_window_px = min(plant_window_px, rm_len)

                    if peak_metric_id == "peak":
                        # use this cm opening within denoise_mask; this has a doubled size extent purge effect
                        row_mask = denoise_mask(row_mask, 0, open_extent_px)

                        # normalize the signal prior to any smoothing so that all
                        # negative values are known to be unsampled.
                        row_mask = row_mask.astype(float)
                        sc_max = row_mask.max()
                        if sc_max:
                            row_mask /= sc_max
                            # sc_max = 1
                        # sc_min = sum_col.min()

                        row_mask = np.sum(row_mask, axis=0)

                        if smoothing_polyorder < row_window_px <= row_mask.size:
                            # smooth the BV curve based on the calculated resolution at certain altitude
                            # smoothing window length is based on the minimal planting density space.
                            # polynomial order is the parameter after testing the different curve smoothing operation.
                            curve_smooth = signal.savgol_filter(row_mask, window_length=row_window_px,
                                                                polyorder=smoothing_polyorder, deriv=0)

                            # Purge lead-in and lead-out negatively projected periods.
                            # This is done to avoid low value projection based compression of the signal range
                            # which greatly affect how the 0.3 peak threshold is imposed upon the resulting curve.
                            for pi in range(len(curve_smooth)):
                                if curve_smooth[pi] >= 0:
                                    break
                                curve_smooth[pi] = 0
                            for pi in range(1, len(curve_smooth) + 1):
                                if curve_smooth[-pi] >= 0:
                                    break
                                curve_smooth[-pi] = 0
                        else:
                            # very short signal or win length, try to process without smoothing since the above call
                            # requirements cannot be met; this is a very unusual situation which may be the result of
                            # an odd geometry
                            tile_obj.mprint(f"WARNING - Row {r + 1} peak smoothing skipped; unsupported input size"
                                            f" [{row_mask.size}] or window length [{row_window_px}] detected.")
                            curve_smooth = row_mask.copy()

                        # find the peaks of the curve
                        peak_x = peakutils.indexes(curve_smooth, min_dist=min_peak_interval, thres=peakutils_thresh)

                        peak_y = np.zeros_like(peak_x)
                        peak_y[:] = row_idx[r]
                    else:
                        max_plants_expected = natural_round(row_mask.shape[1] / row_window_px * overplanting_factor)

                        row_mask = np.asarray(row_mask > 0).astype('uint8')

                        row_mask_dist = cv2.distanceTransform(row_mask, cv2.DIST_L2, 3)
                        label_cnt, label_img, bbox_area, centroids = \
                            cv2.connectedComponentsWithStats(row_mask, connectivity=8)

                        for oi in range(label_cnt):
                            bba = bbox_area[oi]
                            # fill_percentage = bba[4] / (bba[2] * bba[3])
                            if bba[4] < area_thresh_px:
                                label_img[label_img == oi] = 0

                        peaks_keep_x = []
                        peaks_keep_y = []
                        for oi in range(1, label_cnt):
                            # TODO? - use bounding box of each object to propose a target object count for it
                            # less than 1.5 wide but not tall = likely only one plant
                            # tall to short extent ratio could suggest leaf extents (body at each height spike)

                            # for each object, directly assess its x axis peak list to
                            # avoid interference from other objects
                            oi_dist_img = row_mask_dist * (label_img == oi)
                            oi_dist_img_max = np.max(oi_dist_img, axis=0)  # find the peak response of each column
                            peak_x = np.where(oi_dist_img_max > 0)[0]
                            if len(peak_x) == 0:
                                continue

                            peak_y = np.zeros_like(peak_x)
                            peak_y[:] = row_idx[r]
                            # for any repeating x max along y, pick the y position which
                            # is closest to the row center line
                            for pi, pv in enumerate(peak_x):
                                mlist = row_extents[row_key_list[r]][0] + \
                                        np.where(oi_dist_img[:, pv] == oi_dist_img[:, pv].max())[0]
                                peak_yi = row_idx[r]
                                peak_cmin = np.inf
                                for mi in range(len(mlist)):
                                    c_dist = np.abs(mlist[mi] - row_idx[r])
                                    if c_dist < peak_cmin:
                                        peak_cmin = c_dist
                                        peak_yi = mlist[mi]
                                peak_y[pi] = peak_yi

                            o_peaks = oi_dist_img_max[peak_x].copy()
                            o_sort = np.argsort(o_peaks)
                            o_peaks = o_peaks[o_sort]
                            o_x = peak_x[o_sort]
                            o_y = peak_y[o_sort]
                            while True:
                                # pick the most row centered of a set of matching maximums
                                where_max = np.where(o_peaks == o_peaks[-1])[0]
                                if len(where_max) > 1:
                                    c_dist = np.abs(o_y[where_max] - row_idx[r])
                                    where_cmin = np.where(c_dist == c_dist.min())[0]
                                    if len(where_cmin) > 1:
                                        max_idx = where_max[int(natural_round((len(where_cmin) + 1) / 2) - 1)]
                                    else:
                                        max_idx = where_max[where_cmin[0]]
                                else:
                                    max_idx = where_max[0]
                                max_peak = o_peaks[max_idx]
                                max_x = o_x[max_idx]
                                max_y = o_y[max_idx]

                                # add a peak for this current reference point
                                peaks_keep_x.append(max_x)
                                peaks_keep_y.append(max_y)

                                # for the distance comparisons, the y-axis is omitted to avoid top to bottom duplicates
                                x_dist = np.abs(np.asarray(o_x) - max_x)

                                # permit retention of peaks within a configured range of IPS when they are
                                # similar enough to the current peak
                                ppx_s = row_window_px * object_scale_bias_ips_distance_factor[0]
                                ppx_e = row_window_px * object_scale_bias_ips_distance_factor[1]
                                keep_oi = (o_peaks > (max_peak * (1 - ((x_dist / ppx_e) * object_scale_bias_peak_drop_factor)))) & \
                                          (x_dist >= ppx_s) & (x_dist < ppx_e)

                                # retain peaks > the scaled IPS range while discarding any below
                                # the configured percentage of the current peak
                                keep_oi = (keep_oi | (x_dist >= ppx_e)) & \
                                          (o_peaks >= (max_peak * same_object_peak_thresh_factor))

                                keep_oi = np.where(keep_oi)[0]
                                if not len(keep_oi):
                                    # all object peak points have now been excluded or accepted
                                    break

                                # drop any excluded peaks from the next iteration's consideration
                                o_peaks = o_peaks[keep_oi]
                                o_x = o_x[keep_oi]
                                o_y = o_y[keep_oi]

                        peak_x = peaks_keep_x
                        peak_y = peaks_keep_y

                        o_sort = np.argsort(peak_x)
                        peak_x = np.asarray(peak_x)[o_sort]
                        peak_y = np.asarray(peak_y)[o_sort]

                        # # EXPERIMENTAL second pass peak filtering
                        # peak_oi = np.zeros_like(peak_x, float)
                        # for pi in range(len(peak_x)):
                        #     peak_oi[pi] = label_img[peak_y[pi] - row_extents[row_key_list[r]][0], peak_x[pi]]
                        # close_pixels = int(natural_round(0.5 / (2.54 / ppi_w)))
                        # obj_mask = np.zeros_like(row_mask)
                        # for oi in peak_oi:
                        #     obj_mask[label_img == oi] = 1
                        # row_mask_dilated = cv2.dilate(obj_mask, None, iterations=close_pixels)
                        # label_img_dilated = cv2.connectedComponentsWithStats(row_mask_dilated, connectivity=8)[1]
                        # peak_oi_dilated = np.zeros_like(peak_x, float)
                        # for pi in range(len(peak_x)):
                        #     peak_oi_dilated[pi] = label_img_dilated[peak_y[pi] - row_extents[row_key_list[r]][0], peak_x[pi]]
                        # label_img_undilated = np.zeros_like(label_img)
                        # for oi, ov in enumerate(peak_oi):
                        #     label_img_undilated[label_img == ov] = peak_oi_dilated[oi]
                        # label_img = label_img_undilated
                        #
                        # peak_x_kept = peak_x.copy()
                        # peak_y_kept = peak_y.copy()
                        # peaks_keep_x = []
                        # peaks_keep_y = []
                        # for oi in np.unique(peak_oi_dilated):
                        #     keep_xy = np.where(peak_oi_dilated == oi)[0]
                        #     peak_x = peak_x_kept[keep_xy]
                        #     peak_y = peak_y_kept[keep_xy]
                        #
                        #     o_peaks = oi_dist_img_max[peak_x].copy()
                        #     o_sort = np.argsort(o_peaks)
                        #     o_peaks = o_peaks[o_sort]
                        #     o_x = peak_x[o_sort]
                        #     o_y = peak_y[o_sort]
                        #     while True:
                        #         # pick the most row centered of a set of matching maximums
                        #         where_max = np.where(o_peaks == o_peaks[-1])[0]
                        #         if len(where_max) > 1:
                        #             c_dist = np.abs(o_y[where_max] - row_idx[r])
                        #             where_cmin = np.where(c_dist == c_dist.min())[0]
                        #             if len(where_cmin) > 1:
                        #                 max_idx = where_max[int(natural_round((len(where_cmin) + 1) / 2) - 1)]
                        #             else:
                        #                 max_idx = where_max[where_cmin[0]]
                        #         else:
                        #             max_idx = where_max[0]
                        #         max_peak = o_peaks[max_idx]
                        #         max_x = o_x[max_idx]
                        #         max_y = o_y[max_idx]
                        #
                        #         # add a peak for this current reference point
                        #         peaks_keep_x.append(max_x)
                        #         peaks_keep_y.append(max_y)
                        #
                        #         # for the distance comparisons, the y-axis is omitted to avoid top to bottom duplicates
                        #         x_dist = np.abs(np.asarray(o_x) - max_x)
                        #
                        #         # permit retention of peaks within a configured range of IPS when they are
                        #         # similar enough to the current peak
                        #         ppx_s = row_window_px * object_scale_bias_ips_distance_factor[0]
                        #         ppx_e = row_window_px * object_scale_bias_ips_distance_factor[1]
                        #         keep_oi = (o_peaks > (max_peak * (1 - ((x_dist / ppx_e) * object_scale_bias_peak_drop_factor)))) & \
                        #                   (x_dist >= ppx_s) & (x_dist < ppx_e)
                        #
                        #         # retain peaks > the scaled IPS range while discarding any below
                        #         # the configured percentage of the current peak
                        #         keep_oi = (keep_oi | (x_dist >= ppx_e)) & \
                        #                   (o_peaks >= (max_peak * same_object_peak_thresh_factor))
                        #
                        #         keep_oi = np.where(keep_oi)[0]
                        #         if not len(keep_oi):
                        #             # all object peak points have now been excluded or accepted
                        #             break
                        #
                        #         # drop any excluded peaks from the next iteration's consideration
                        #         o_peaks = o_peaks[keep_oi]
                        #         o_x = o_x[keep_oi]
                        #         o_y = o_y[keep_oi]
                        #
                        # peak_x = peaks_keep_x
                        # peak_y = peaks_keep_y
                        #
                        # o_sort = np.argsort(peak_x)
                        # peak_x = np.asarray(peak_x)[o_sort]
                        # peak_y = np.asarray(peak_y)[o_sort]

                        excess_plants = len(peak_x) - max_plants_expected

                        # This filtering effectively relaxes the peak picking, allowing it to lightly
                        # bias toward over-counts.
                        # That must not be allowed to turn actual low counts into forced full counts.

                        if excess_plants > 0:
                            row_constrained_list.append(r + 1)
                            row_mask_dist_max = np.max(row_mask_dist, axis=0)

                        while excess_plants > 0:
                            # Goal seek the count until it drops below an upper constraint by IPS.

                            # Order the x/y positions according to their nearest neighbor distance.
                            # Drop whichever one has the least distance to any neighbor.
                            # Repeat until reaching the max acceptable count.

                            all_dist = np.zeros((len(peak_x), len(peak_x)), float)
                            for oi in range(len(peak_x)):
                                for ni in range(len(peak_x)):
                                    all_dist[oi, ni] = np.sqrt((peak_x[oi] - peak_x[ni]) ** 2 +
                                                               (peak_y[oi] - peak_y[ni]) ** 2)

                            # It would be better if each were scored for selection by a more robust metric.
                            drop_i = np.where(all_dist == all_dist[all_dist > 0].min())[0]

                            if len(drop_i) > 1:
                                # keep only the minimums that are all related to the first
                                drop_dist = np.zeros_like(drop_i, float)
                                for di, dv in enumerate(drop_i):
                                    drop_dist[di] = np.sqrt((peak_x[drop_i[0]] - peak_x[dv]) ** 2 +
                                                            (peak_y[drop_i[0]] - peak_y[dv]) ** 2)
                                drop_i = np.append(drop_i[0],
                                                   drop_i[np.where(drop_dist == drop_dist[drop_dist > 0].min())[0]])

                            if len(drop_i) > 1:
                                # Bias against dropping a single peak object in the presence of any other with multiple.
                                # At least one compliment will be seen in every case.
                                peak_oi = np.zeros_like(peak_x, float)
                                for pi in range(len(peak_x)):
                                    peak_oi[pi] = label_img[peak_y[pi] - row_extents[row_key_list[r]][0], peak_x[pi]]

                                # # sort the drop candidates by distance from center max to min so that a max
                                # # offset point is the default drop by being listed first
                                # c_dist = np.abs(peak_y[drop_i] - row_idx[r])
                                # drop_i = drop_i[np.argsort(c_dist)][::-1]

                                # sort the drop candidates by distance from object edge min to max so that a min
                                # distance point is the default drop by being listed first
                                drop_dist = row_mask_dist_max[peak_x[drop_i]]
                                drop_i = drop_i[np.argsort(drop_dist)]

                                if len(np.unique(peak_oi[drop_i])) == 1:
                                    # all peaks are on the same object; drop the first
                                    drop_i = [drop_i[0]]
                                else:
                                    # the peaks are on different objects; try to keep single peak objects
                                    peak_oi_cnt = np.zeros_like(peak_x, int)
                                    for pi in range(len(peak_x)):
                                        peak_oi_cnt[pi] = np.sum(peak_oi == peak_oi[pi])
                                    for di in drop_i:
                                        if peak_oi_cnt[di] > 1:
                                            # drop from the first object with multiple peaks
                                            drop_i = [di]
                                            break
                                    if len(drop_i) > 1:
                                        # all candidates are on a single peak object
                                        # Biasing where distance map values decide which to keep.
                                        # Pick the lowest valued distance (the thinnest object) to drop.
                                        y_offset = row_extents[row_key_list[r]][0]
                                        min_di = np.inf
                                        select_di = 0  # use first by default
                                        for di in drop_i:
                                            row_y = int(natural_round(peak_y[di] - y_offset))
                                            di_mask_dist = row_mask_dist[row_y, peak_x[di]]
                                            if di_mask_dist < min_di:
                                                min_di = di_mask_dist
                                                select_di = di
                                        drop_i = [select_di]

                            keep_i = np.arange(0, len(peak_x)) != drop_i[0]
                            peak_x = peak_x[keep_i]
                            peak_y = peak_y[keep_i]

                            excess_plants -= 1

                    # the number of peaks is the estimation of the plants
                    num_peaks = len(peak_x)
                    # tile_obj.mprint(num_peaks)

                    stand_row[r] = num_peaks

                    # TODO - add lat/lon for each; problem: the transform/scaled_transform origin is not rotated
                    if num_peaks > 0:
                        # Generate the CSV output records for this row
                        row_stand_xin = []
                        row_stand_yin = []
                        row_zero_xpx = 0  # peak_x is already relative to object_row_sei[0]
                        row_zero_ypx = row_idx[r] - (meta_row_spacing * ppi_w / 2)
                        for p in range(num_peaks):
                            x_cm = yx_cm[1] * (peak_x[p] - row_zero_xpx)
                            y_cm = yx_cm[0] * (peak_y[p] - row_zero_ypx)
                            row_stand_xin.append(x_cm / 2.54)
                            row_stand_yin.append(y_cm / 2.54)
                            plant_loc = (row_sei[0] + peak_x[p], peak_y[p])  # row_idx[r])
                            output_line_entry = f"{peak_metric_id}-stand,{plot_id},{rinfo['ccw_alignment_rotation']}," \
                                                f"{round(x_cm, 6)},{round(y_cm, 6)},{r + 1},{is_harv_row[r]}," \
                                                f"{round(plant_loc[0])},{round(plant_loc[1])}"
                            plant_loc_table.append(output_line_entry)
                        row_stand_xin_byplant[r] = row_stand_xin
                        row_stand_yin_byplant[r] = row_stand_yin

                    # do not add count to total count if it is not a harvest row
                    if is_harv_row[r]:
                        # The number of the plants
                        num_plant += num_peaks
                        if num_peaks > 0:
                            # Number of rows on which stand was computed
                            # Exclude empty rows so that QC may consider this
                            num_row += 1

            if len(row_constrained_list) > 0:
                add_reason = f"Count constrained to seeding count for row|s: {row_constrained_list}."
                tile_obj.mprint(f"WARNING - ({peak_metric_id}) {add_reason}")
                if peak_metric_id == "peak":
                    if len(object_stand_reason):
                        peak_stand_reason = f"{peak_stand_reason}; {add_reason}"
                    else:
                        peak_stand_reason = add_reason
                else:
                    if len(object_stand_reason):
                        object_stand_reason = f"{object_stand_reason}; {add_reason}"
                    else:
                        object_stand_reason = add_reason

            if meta_harv_row != num_row:
                add_reason = f"Number of harvest rows with stand: {num_row}; expected {meta_harv_row}."
                tile_obj.mprint(f"WARNING - ({peak_metric_id}) {add_reason}")
                if peak_metric_id == "peak":
                    if len(object_stand_reason):
                        peak_stand_reason = f"{peak_stand_reason}; {add_reason}"
                    else:
                        peak_stand_reason = add_reason
                else:
                    if len(object_stand_reason):
                        object_stand_reason = f"{object_stand_reason}; {add_reason}"
                    else:
                        object_stand_reason = add_reason

            if num_row == 0:
                num_plant = None

            if not np.all(np.array(is_harv_row)):
                hstand_row = np.array(stand_row)
                hstand_row = hstand_row[np.where(is_harv_row)[0].astype(int)]
                tile_obj.mprint(f"row_{peak_metric_id}_stand: {stand_row};"
                                f" harvest_row_{peak_metric_id}_stand: {hstand_row}")
            else:
                tile_obj.mprint(f"row_{peak_metric_id}_stand: {stand_row}")

            if peak_metric_id == "peak":
                this_stand_results = {"peak_stand": copy(num_plant), "row_peak_stand": copy(stand_row),
                                      "row_peak_stand_xin_byplant": safefloat(copy(row_stand_xin_byplant)),
                                      "row_peak_stand_yin_byplant": safefloat(copy(row_stand_yin_byplant)),
                                      "peak_stand_reason": copy(peak_stand_reason)}
                row_sei = peak_row_sei
            else:
                this_stand_results = {"object_stand": copy(num_plant), "row_object_stand": copy(stand_row),
                                      "row_object_stand_xin_byplant": safefloat(copy(row_stand_xin_byplant)),
                                      "row_object_stand_yin_byplant": safefloat(copy(row_stand_yin_byplant)),
                                      "object_stand_reason": copy(object_stand_reason)}
                row_sei = object_row_sei

            # labeling plants on the plot label image
            if plant_labeling is True or call_ka:
                mask_color = add_row_labels(rot_mask_img.astype(np.uint8).copy(), row_idx,
                                            row_extents, is_harv_row=is_harv_row, row_sei=row_sei)

                l_rad = int(np.ceil(1.5 * ppi_l))
                for k in range(1, len(plant_loc_table)):
                    if plant_loc_table[k].split(',')[0] == f"{peak_metric_id}-stand":
                        plant_loc = (int(natural_round(float(plant_loc_table[k].split(',')[7]))),
                                     int(natural_round(float(plant_loc_table[k].split(',')[8]))))
                        if plant_loc_table[k].split(',')[4] == 'True':
                            # metric row
                            lcolor = (0, 255, 0)
                        else:
                            # non-metric row
                            lcolor = (255, 255, 0)

                        cv2.circle(mask_color, plant_loc, radius=l_rad, color=lcolor, thickness=1, lineType=8, shift=0)

                # reverse the rotation applied and write using the input transform
                rev_rotation = -(heading - 90) + 90
                rot_mask_color = rotate_plot(mask_color, alpha, rev_rotation)
                rot_alpha = rotate_plot(alpha, alpha, rev_rotation)

                # add additional meta and save the resulting raster
                rot_img_rio = np.zeros((4, rot_mask_color.shape[0], rot_mask_color.shape[1]), 'uint8')
                rot_img_rio[0, ...] = rot_mask_color[..., 0]
                rot_img_rio[1, ...] = rot_mask_color[..., 1]
                rot_img_rio[2, ...] = rot_mask_color[..., 2]
                del rot_mask_color
                rot_img_rio[3, ...] = rot_alpha

                stand_path = f"{plot_id}_{peak_metric_id}_stand_label.tif"
                gdal_dict_out = gdal_open(file_name=stand_path, mode='CREATE',
                                          create_params={'data_type': rot_img_rio.dtype, 'shape': rot_img_rio.shape},
                                          add_alpha=False)

                this_metadata = deepcopy(tile_obj.gdal_dict_ref['metadata'])
                this_metadata.update({"stand_results": json.dumps(this_stand_results)})

                gdal_dict_out.update({'image_data': rot_img_rio,
                                      'transform': rinfo['scaled_tform'],
                                      'band_color_interps': [3, 4, 5, 6],
                                      'band_descriptions': ['red', 'green', 'blue', 'alpha'],
                                      'projection': tile_obj.gdal_dict_ref['projection'],
                                      'metadata': this_metadata})
                gdal_update(gdal_dict=gdal_dict_out, band_stats=True)
                del rot_img_rio

                gdal_close(gdal_dict=gdal_dict_out)

                if call_ka:
                    keep_artifact(stand_path, os.path.basename(stand_path), plot_id)

            stand_results.update(this_stand_results)

    if len(stand_reason):
        if len(peak_stand_reason):
            peak_stand_reason = f"{stand_reason}; {peak_stand_reason}"
        else:
            peak_stand_reason = stand_reason
        if len(object_stand_reason):
            object_stand_reason = f"{stand_reason}; {object_stand_reason}"
        else:
            object_stand_reason = stand_reason
        stand_results.update({"peak_stand_reason": peak_stand_reason, "object_stand_reason": object_stand_reason})

    stand_results.update({"observed_rows": num_row_qc_obs, "counted_rows": num_row})

    deepstand_results = dict()

    if peak_stand_done:
        row_deepstand_xin_byplant = [[None]] * num_row_original
        row_deepstand_yin_byplant = [[None]] * num_row_original
        deepstand_row = [None] * meta_plot_row

        # NOTE - while this may be possible to run without dependency on peak stand, the thought is that
        #        deep stand would not have been given reasonable input calling into question the value
        #        of output it may produce in these odd cases
        if 'sagemaker_response' in plot_properties and plot_properties['sagemaker_response'] is not None:
            # this happens within pipeline since the tile_obj never gets a chance to
            # consolidate anything except the CCVRnet or IDC mask, leaving this response to be stored
            # here just prior to uvastandcount.process being called.
            tile_obj.store_deepstand_result(plot_properties['sagemaker_response'])

        deepstand_output = tile_obj.get_deepstand_output(plot_image=rgb_bands)
        # NOTE - The uavanalytics wrapper logic is not expected to assign anything to deepstand_reason.
        #        However, it may be set by a direct endpoint call.
        deepstand_reason = deepstand_output['deepstand_reason']

        # deepstand_results = dict()
        # deepstand_reason = ""
        # if 'uavdeepstand' not in tile_obj.method_output:
        #     # producing it now; apparently no batch transform output was stored for usage
        #     tile_obj.call_deepstand()
        #
        # if 'uavdeepstand' not in tile_obj.method_output:
        #     tile_obj.mprint("ERROR - deepstand results could not be found/generated.")
        # else:
        #     deepstand_results = tile_obj.method_output['uavdeepstand']
        #     if 'deepstand_reason' in deepstand_results:
        #         deepstand_reason = deepstand_results['deepstand_reason']

        if 'jsdoc' in deepstand_output:
            bboxjs = deepcopy(deepstand_output["jsdoc"])

            num_row_ds = 0
            num_plant_ds = 0

            # 20 is hard coded.. so adding some statistical fuzziness
            # assume a CV of 1 and 20 as mean cy val
            # then std dev = 2 and interquartile range as (18.6,21.3)
            # so cy from top is 22
            boundary_buffer = 22  # TODO? - enforce this as is done with peak stand

            for r in range(num_row_original):
                num_ds = 0

                idx_str = f"00{r + 1}"[-3:]
                extents_ = row_extents[f'row_{idx_str}']
                row_deepstand_xin = []
                row_deepstand_yin = []
                row_zero_px = row_idx[r] - (meta_row_spacing * ppi_w / 2)
                for cell in bboxjs:
                    cx = cell["x1"] + (cell["x2"] - cell["x1"]) / 2
                    cy = cell["y1"] + (cell["y2"] - cell["y1"]) / 2
                    # check for boundary cases
                    if extents_[0] <= cy <= extents_[1] and row_sei[0] <= cx <= row_sei[1]:
                        score = cell['score']
                        x_cm = yx_cm[1] * (cx - row_sei[0])
                        y_cm = yx_cm[0] * (cy - row_zero_px)
                        row_deepstand_xin.append(x_cm / 2.54)
                        row_deepstand_yin.append(y_cm / 2.54)
                        output_line_entry = f"deep-stand,{plot_id},{rinfo['ccw_alignment_rotation']}," \
                                            f"{round(x_cm, 6)},{round(y_cm, 6)},{r + 1},{is_harv_row[r]}," \
                                            f"{int(cx)},{int(cy)},{score}"
                        plant_loc_table.append(output_line_entry)
                        num_ds += 1
                if len(row_deepstand_xin) > 0:
                    row_deepstand_xin_byplant[r] = row_deepstand_xin
                    row_deepstand_yin_byplant[r] = row_deepstand_yin

                if is_harv_row[r]:
                    # The number of the plants
                    num_plant_ds += num_ds
                    if num_ds > 0:
                        # Number of rows on which deep stand was computed
                        # Exclude empty rows so that QC may consider this
                        num_row_ds += 1

                deepstand_row[r] = num_ds

            if not np.all(np.array(is_harv_row)):
                hstand_row = np.array(deepstand_row)
                hstand_row = hstand_row[np.where(is_harv_row)[0].astype(int)]
                tile_obj.mprint(f"row_deepstand: {deepstand_row}; harvest_row_stand: {hstand_row}")
            else:
                tile_obj.mprint(f"row_deepstand: {deepstand_row}")

            if num_row_ds == 0:
                num_plant_ds = None

            deepstand_results.update({"deepstand": num_plant_ds, "row_deepstand": deepstand_row,
                                      "row_deepstand_xin_byplant": safefloat(row_deepstand_xin_byplant),
                                      "row_deepstand_yin_byplant": safefloat(row_deepstand_yin_byplant),
                                      "deepstand_counted_rows": num_row_ds, "deepstand_reason": deepstand_reason})

            # labeling plants on the plot label image
            if plant_labeling is True or call_ka:
                mask_color = add_row_labels(rot_mask_img.astype(np.uint8).copy(), row_idx,
                                            row_extents, is_harv_row=is_harv_row, row_sei=row_sei)

                rgb_label = np.zeros((rgb_bands.shape[0], rgb_bands.shape[1], 4), rgb_bands.dtype)
                rgb_label[..., 0] = rgb_bands[..., 0]
                rgb_label[..., 1] = rgb_bands[..., 1]
                rgb_label[..., 2] = rgb_bands[..., 2]
                rgb_label[..., 3] = alpha

                if rgb_label.dtype.name.lower() in ['uint16', 'int16']:
                    # convert to uint8 for labeled outputs (and JPG requires it)
                    rgb_label[rgb_label < 0] = 0
                    bmax = rgb_label[..., 0:3].max()
                    if bmax > 255:  # skip for any apparent 8bit data hiding within an unusual data type
                        if bmax <= 4095:
                            # very likely to be 12 bit not 16
                            tmax = 4095
                        else:
                            tmax = 65535
                        tmax = min(tmax, bmax)  # pull up, reducing decimation, adding brightness & preserving contrast
                        rgb_label[..., 0:3] = rgb_label[..., 0:3] / tmax * 255

                    rgb_label = rgb_label.astype('uint8')

                rgb_label_csv = rgb_label.copy()

                l_rad = int(np.ceil(1.5 * ppi_l))
                for k in range(1, len(plant_loc_table)):
                    if plant_loc_table[k].split(',')[0] == 'deep-stand':
                        plant_loc = (int(natural_round(float(plant_loc_table[k].split(',')[7]))),
                                     int(natural_round(float(plant_loc_table[k].split(',')[8]))))
                        if plant_loc_table[k].split(',')[4] == 'True':
                            # metric row
                            lcolor = (0, 255, 0)
                        else:
                            # non-metric row
                            lcolor = (255, 255, 0)

                        cv2.circle(mask_color, plant_loc, radius=l_rad, color=lcolor, thickness=1,
                                   lineType=8, shift=0)

                        cv2.circle(rgb_label, plant_loc, radius=l_rad, color=lcolor, thickness=1,
                                   lineType=8, shift=0)

                        cv2.circle(rgb_label_csv, plant_loc, radius=l_rad, color=(0, 255, 255),
                                   thickness=1, lineType=8, shift=0)

                # Write the CSV matched non-rotated label image for support of DL and Figure8 workflows:
                img_rio = np.zeros((3, rgb_label_csv.shape[0], rgb_label_csv.shape[1]), rgb_label_csv.dtype)
                img_rio[0, ...] = rgb_label_csv[..., 0]
                img_rio[1, ...] = rgb_label_csv[..., 1]
                img_rio[2, ...] = rgb_label_csv[..., 2]
                del rgb_label_csv

                jpg_meta = {"driver": "JPEG",
                            "quality": 95,
                            "nodata": None,
                            "dtype": img_rio.dtype,
                            "height": img_rio.shape[1],
                            "width": img_rio.shape[2],
                            "count": img_rio.shape[0]}

                deepstand_rcsv_label_path = f"{plot_id}_deepstand_rgb_csv_label.jpg"
                with rasterio.open(deepstand_rcsv_label_path, 'w', **jpg_meta) as dst:
                    dst.write(img_rio)
                del img_rio, jpg_meta

                # reverse the rotation applied and write using the input transform
                rev_rotation = -(heading - 90) + 90
                rot_rgb_bands = rotate_plot(rgb_label, alpha, rev_rotation)
                del rgb_label
                rot_mask_color = rotate_plot(mask_color, alpha, rev_rotation)
                del mask_color
                rot_alpha = rotate_plot(alpha, alpha, rev_rotation)

                # add additional meta and save the resulting raster
                rot_img_rio = np.zeros((4, rot_mask_color.shape[0], rot_mask_color.shape[1]), 'uint8')
                rot_img_rio[0, ...] = rot_mask_color[..., 0]
                rot_img_rio[1, ...] = rot_mask_color[..., 1]
                rot_img_rio[2, ...] = rot_mask_color[..., 2]
                del rot_mask_color
                rot_img_rio[3, ...] = rot_alpha

                deepstand_mlabel_path = f"{plot_id}_deepstand_label.tif"
                gdal_dict_out = gdal_open(file_name=deepstand_mlabel_path, mode='CREATE',
                                          create_params={'data_type': rot_img_rio.dtype, 'shape': rot_img_rio.shape},
                                          add_alpha=False)

                this_metadata = deepcopy(tile_obj.gdal_dict_ref['metadata'])
                this_metadata.update({"deepstand_results": json.dumps(deepstand_results)})

                gdal_dict_out.update({'image_data': rot_img_rio,
                                      'transform': rinfo['scaled_tform'],
                                      'band_color_interps': [3, 4, 5, 6],
                                      'band_descriptions': ['red', 'green', 'blue', 'alpha'],
                                      'projection': tile_obj.gdal_dict_ref['projection'],
                                      'metadata': this_metadata})
                gdal_update(gdal_dict=gdal_dict_out, band_stats=True)
                del rot_img_rio

                # add additional meta and save the resulting raster
                rot_img_rio = np.zeros((4, rot_rgb_bands.shape[0], rot_rgb_bands.shape[1]), 'uint8')
                rot_img_rio[0, ...] = rot_rgb_bands[..., 0]
                rot_img_rio[1, ...] = rot_rgb_bands[..., 1]
                rot_img_rio[2, ...] = rot_rgb_bands[..., 2]
                del rot_rgb_bands
                rot_img_rio[3, ...] = rot_alpha
                del rot_alpha

                deepstand_rlabel_path = f"{plot_id}_deepstand_rgb_label.tif"
                gdal_dict_out = gdal_open(file_name=deepstand_rlabel_path, mode='CREATE',
                                          create_params={'data_type': rot_img_rio.dtype, 'shape': rot_img_rio.shape},
                                          add_alpha=False)

                gdal_dict_out.update({'image_data': rot_img_rio,
                                      'transform': rinfo['scaled_tform'],
                                      'band_color_interps': [3, 4, 5, 6],
                                      'band_descriptions': ['red', 'green', 'blue', 'alpha'],
                                      'projection': tile_obj.gdal_dict_ref['projection'],
                                      'metadata': tile_obj.gdal_dict_ref['metadata']})
                gdal_update(gdal_dict=gdal_dict_out, band_stats=True)
                del rot_img_rio

                if call_ka:
                    keep_artifact(deepstand_mlabel_path, os.path.basename(deepstand_mlabel_path), plot_id)
                    keep_artifact(deepstand_rlabel_path, os.path.basename(deepstand_rlabel_path), plot_id)
                    keep_artifact(deepstand_rcsv_label_path, os.path.basename(deepstand_rcsv_label_path), plot_id)

    stand_results.update(deepstand_results)

    # output the plant location table
    if plant_loc_csv is True or call_ka:
        plant_loc_path = f"{plot_id}_plant_location.csv"
        with open(plant_loc_path, 'w') as csv_table:
            for item in plant_loc_table:
                csv_table.write('%s\n' % item)

        if call_ka:
            keep_artifact(plant_loc_path, os.path.basename(plant_loc_path), plot_id)

    return stand_results, plant_loc_table, mask_img
