"""
UAV Stand Counting Module

This module detects and counts individual stands (e.g., plants) in UAV-captured RGB images. 
It leverages image processing techniques, including adaptive thresholding and contour 
detection, to identify and count objects of interest based on size.

Explanation of Code Structure and Steps
Preprocess the Image:

Converts the input RGB image to grayscale.
Uses Gaussian blur to smooth out noise and improve edge detection.
Applies adaptive thresholding, which dynamically adjusts based on local pixel values, making it resilient to lighting variations. The output is a binary image.
Detect Objects Using Contours:

Performs morphological operations to close small gaps and better define contours.
Finds contours in the binary image and filters them based on area, ignoring too-small or too-large objects based on defined thresholds in config.
Count Stands:

Counts the number of contours that meet the specified size requirements.
Returns the final count along with details for further inspection.
Configuration:

Stores size thresholds (min_area and max_area) for easy adjustments, allowing the script to be adaptable to different object sizes and densities.
"""

import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesses the input image by converting it to grayscale, applying Gaussian blur, 
    and using adaptive thresholding for binarization.
    
    Args:
        image (numpy.ndarray): Input RGB image from UAV.
    
    Returns:
        numpy.ndarray: Binary image suitable for contour-based object detection.
    """
    # Convert to grayscale for simpler processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and enhance object separation
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    
    # Use adaptive thresholding to handle variable lighting conditions
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    return binary_image

def detect_objects(binary_image, config):
    """
    Detects and counts objects in the binary image using contour detection and size filtering.
    
    Args:
        binary_image (numpy.ndarray): The binary preprocessed image.
        config (dict): Configuration parameters containing size thresholds for objects.
    
    Returns:
        int: Count of detected objects meeting the specified size criteria.
    """
    # Apply morphological operations to close small holes and separate close objects
    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the morphed binary image
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_count = 0
    for contour in contours:
        # Calculate area of each contour to filter out noise and irrelevant objects
        area = cv2.contourArea(contour)
        
        # Only count objects within the specified area range
        if config["min_area"] < area < config["max_area"]:
            object_count += 1

    return object_count

def count_stands(image, config):
    """
    Main function to count stands in the input image using preprocessing and detection.
    
    Args:
        image (numpy.ndarray): Input UAV RGB image.
        config (dict): Configuration dictionary with thresholds for processing and filtering.
    
    Returns:
        dict: Dictionary containing the stand count and additional details for diagnostics.
    """
    # Step 1: Preprocess the image to obtain a binary image
    binary_image = preprocess_image(image)
    
    # Step 2: Detect and count objects in the binary image
    stand_count = detect_objects(binary_image, config)
    
    return {
        "stand_count": stand_count,
        "details": "Stand count obtained using adaptive thresholding and contour-based detection with size filtering."
    }

# Configuration dictionary with thresholds for size-based filtering
config = {
    "min_area": 100,    # Minimum area of objects to consider as valid stands
    "max_area": 5000    # Maximum area of objects to consider as valid stands
}

# Example wrapper function to process the UAV image tile object
def process_image(tile_obj):
    """
    Wrapper function to process an input image object from UAV and compute stand count.
    
    Args:
        tile_obj: Object containing the image data and configuration parameters.
    
    Returns:
        dict: Dictionary containing the stand count result and additional information.
    """
    # Extract the rotated and oriented image for processing
    plot_image = tile_obj.get_rotate_by_heading_output(reprocess=False)[0]
    
    # Perform stand counting
    return count_stands(plot_image, config)
