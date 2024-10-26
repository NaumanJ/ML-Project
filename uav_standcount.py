# --------------------------------------------------------
# Name: CornStandCount
# Purpose: Counting the corn stands using plot-level image.
#
# --------------------------------------------------------

"""
This module performs stand counting using UAV RGB images.
The code detects and counts distinct objects (e.g., plants) based on contour detection.

Explanation of Code Sections:
Configuration and Initialization: A configuration dictionary, config, specifies minimum and maximum areas for valid objects to filter out noise and irrelevant objects.

Image Preprocessing:

Converts the image to grayscale.
Applies Gaussian blur to smooth out the image and reduce noise.
Uses binary thresholding to create a binary mask where objects are distinct from the background.
Object Detection Using Contours:

Finds contours (object boundaries) in the binary image.
Filters contours based on area (using min_area and max_area) to count only objects that match the expected size of stands or plants.
Main Counting Function:

Combines preprocessing and object detection steps to get the final count of valid objects.
Returns the result, including a count of detected stands and diagnostic information.
"""

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesses the input image by converting to grayscale, applying Gaussian blur, and using binary thresholding.
    
    Args:
        image (numpy.ndarray): The input RGB image.
    
    Returns:
        numpy.ndarray: Preprocessed binary image suitable for contour detection.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise and enhance object edges
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply binary thresholding to create a binary image for contour detection
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    
    return binary_image

def detect_objects(binary_image, config):
    """
    Detects objects in the binary image using contour detection, then filters based on size.
    
    Args:
        binary_image (numpy.ndarray): The preprocessed binary image.
        config (dict): Configuration dictionary containing object size thresholds.
    
    Returns:
        int: Count of detected objects that meet the specified size criteria.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_count = 0
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)
        
        # Filter based on area to exclude noise and non-relevant objects
        if config["min_area"] < area < config["max_area"]:
            object_count += 1

    return object_count

def count_stands(image, config):
    """
    Main function to count objects in the image using preprocessing and detection functions.
    
    Args:
        image (numpy.ndarray): The input UAV RGB image.
        config (dict): Configuration dictionary containing parameters for processing and filtering.
    
    Returns:
        dict: Result containing the count of detected stands and additional info.
    """
    # Preprocess the image to get a binary image for contour detection
    binary_image = preprocess_image(image)
    
    # Detect and count objects in the binary image
    stand_count = detect_objects(binary_image, config)
    
    return {
        "stand_count": stand_count,
        "details": "Stand counting completed using contour detection with size filtering."
    }

# Example configuration with size thresholds for objects
config = {
    "min_area": 50,     # Minimum contour area to consider as a valid object
    "max_area": 5000    # Maximum contour area to consider as a valid object
}


# Example usage function
def process_image(tile_obj):
    """
    Wrapper function to process an input image object from UAV and count stands.
    
    Args:
        tile_obj: Object containing image data.
    
    Returns:
        dict: Dictionary containing the stand count result.
    """
    # Extract image data
    plot_image = tile_obj.get_rotate_by_heading_output(reprocess=False)[0]

    # Perform stand counting and return result
    return count_stands(plot_image, config)
