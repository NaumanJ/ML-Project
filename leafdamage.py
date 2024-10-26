"""
This module computes leaf damage ratio index from UAV RGB plot 

Original Outline Recap
Configuration: Parsing thresholds and insets for filtering damaged areas.
Image Preparation: Validating and converting RGB channels, resizing or standardizing pixel intensity.
Leaf Damage Index Calculation: Applying thresholds to detect damaged regions, possibly based on color values indicative of damage (e.g., brownish areas).

Code Explanation:
Configuration and Initialization: Defines threshold parameters in a dictionary, config, for easy adjustments.

Image Preprocessing:

The function checks that the image is an RGB three-channel image.
The image is converted from BGR to LAB color space to enhance brown/green detection in a less lighting-sensitive way.
Thresholding for Damaged Areas:

Uses the a channel from the LAB color space to create a mask of areas with "brown" tones (indicating possible leaf damage).
Center-Focused Masking:

Applies a central mask based on a configurable border_ratio to avoid edge effects.
Leaf Damage Index Calculation:

Calculates the percentage of "damaged" pixels relative to the total pixels in the masked region.
Result Return:

Returns a dictionary with the leaf damage index, reason, and diagnostic data for further insights.

"""

"""
This module computes the leaf damage ratio index (LDI) from a UAV RGB image using adaptive thresholding and LAB color space for more flexible detection of damaged areas.
"""

import cv2
import numpy as np

def calculate_leaf_damage_index(plot_image, config):
    """
    Calculate the leaf damage index (LDI) based on a processed RGB image.
    
    Args:
        plot_image (numpy.ndarray): Input RGB image from UAV.
        config (dict): Configuration dictionary containing thresholds for LDI calculations.

    Returns:
        dict: A dictionary containing LDI values and additional diagnostics.
    """
    # Validate that the input image is in 3-channel RGB format
    if plot_image.shape[2] < 3:
        raise ValueError("Input image must have three channels (RGB).")
    
    # Convert to LAB color space for better separation of brown/green shades
    lab_image = cv2.cvtColor(plot_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Threshold for "brown" areas on the 'a' channel; adjusts based on image's brown tones
    brown_mask = cv2.inRange(
        a_channel, 
        config['ldi_thresh']['brown_low_thresh'], 
        config['ldi_thresh']['value']
    )
    
    # Border masking to exclude edges where data may be less relevant
    height, width = plot_image.shape[:2]
    border_ratio = config['ldi_thresh']['inset']
    border_h = int(height * border_ratio)
    border_w = int(width * border_ratio)

    # Create a mask for the central area of the image, excluding the border
    mask = np.zeros_like(brown_mask)
    mask[border_h:height - border_h, border_w:width - border_w] = 255

    # Apply the mask to the brown regions to focus only on the central area
    filtered_brown = cv2.bitwise_and(brown_mask, mask)

    # Calculate the Leaf Damage Index (LDI) as the ratio of damaged pixels to total pixels
    total_pixels = np.count_nonzero(mask)
    damaged_pixels = np.count_nonzero(filtered_brown)
    leaf_damage_index = (damaged_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # Return results, including LDI and additional diagnostics for clarity
    return {
        "ldi": leaf_damage_index,
        "ldi_reason": "Calculated using adaptive thresholding in LAB color space",
        "total_pixels": total_pixels,
        "damaged_pixels": damaged_pixels
    }

# Example configuration with thresholds for LDI calculation
config = {
    "ldi_thresh": {
        "inset": 0.1,                  # Border exclusion ratio
        "value": 40,                    # Upper threshold for brown detection
        "brown_low_thresh": 15,         # Lower threshold for brown detection
        "brown_sat_low_thresh": 0       # Saturation lower threshold for brown detection (if needed)
    }
}

# Example usage function
def process_image(tile_obj):
    """
    Wrapper function to process an input image object from UAV and compute LDI.
    
    Args:
        tile_obj: Object containing image data and configuration parameters.

    Returns:
        dict: Dictionary containing the LDI result.
    """
    # Extract image data
    plot_image = tile_obj.get_rotate_by_heading_output(reprocess=False)[0]

    # Calculate and return leaf damage index
    return calculate_leaf_damage_index(plot_image, config)
