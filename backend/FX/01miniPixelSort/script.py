# script.py

import cv2
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Configure logger if it hasn't been configured yet
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_param(value, default, param_type, min_val=None, max_val=None):
    """
    Validate and return the parameter value.
    """
    try:
        value = param_type(value)
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid value: {value}. Using default: {default}")
        value = default
    return value

def rotate_image(image, angle, center=None):
    """
    Rotate the image by the specified angle around the given center.
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    logger.debug(f"Rotated image by {angle} degrees.")
    return rotated_image

def apply_effect(
    frame,
    mask=None,
    threshold_min=0,
    threshold_max=255,
    sorting_style=0,
    angle=0.0,
    contrast=1.0,
    saturation=1.0,
    resolution=1,
    noise_amount=0,
    noise_scale=1.0,
    sorting_direction=0,
    opacity=1.0
):
    """
    Apply the pixel sorting effect to the given frame within the specified mask.
    """
    # Log the received parameters
    logger.debug(f"Received Parameters - Threshold Min: {threshold_min}, Threshold Max: {threshold_max}, "
                 f"Sorting Style: {sorting_style}, Angle: {angle}, Contrast: {contrast}, "
                 f"Saturation: {saturation}, Resolution: {resolution}, Noise Amount: {noise_amount}, "
                 f"Noise Scale: {noise_scale}, Sorting Direction: {sorting_direction}, "
                 f"Opacity: {opacity}")
    
    # Adjust threshold ranges based on sorting_style
    if sorting_style == 0:  # Hue
        thresh_min = get_param(threshold_min, 0, int, 0, 179)
        thresh_max = get_param(threshold_max, 179, int, 0, 179)
        logger.debug(f"Adjusted Thresholds for Hue - Min: {thresh_min}, Max: {thresh_max}")
    elif sorting_style == 1:  # Saturation
        thresh_min = get_param(threshold_min, 0, int, 0, 255)
        thresh_max = get_param(threshold_max, 255, int, 0, 255)
        logger.debug(f"Adjusted Thresholds for Saturation - Min: {thresh_min}, Max: {thresh_max}")
    else:  # Lightness
        thresh_min = get_param(threshold_min, 0, int, 0, 255)
        thresh_max = get_param(threshold_max, 255, int, 0, 255)
        logger.debug(f"Adjusted Thresholds for Lightness - Min: {thresh_min}, Max: {thresh_max}")

    # Apply rotation to the frame if needed
    if angle != 0:
        frame = rotate_image(frame, angle)

    def pixel_sort(image, mask, axis, style, thresh_min, thresh_max):
        """
        Perform pixel sorting on the image within the mask based on the specified channel.
        """
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0]
            logger.debug("Sorting by Hue.")
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1]
            logger.debug("Sorting by Saturation.")
        else:  # Sort by Lightness (brightness)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale
            logger.debug("Sorting by Lightness.")

        # Log channel statistics
        logger.debug(f"Channel to sort - min: {channel_to_sort.min()}, max: {channel_to_sort.max()}")

        # Combine pixel intensity thresholding and mask
        pixel_mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max) & mask
        num_pixels = np.sum(pixel_mask)
        logger.debug(f"Number of pixels passing the mask and threshold: {num_pixels}")

        if num_pixels == 0:
            logger.warning("No pixels within the mask are passing the threshold check!")

        # Ensure the mask is strictly respected
        sorted_image = image.copy()
        if axis == 0:  # Vertical sorting
            logger.debug("Sorting vertically.")
            for col in range(image.shape[1]):
                masked_indices = np.where(pixel_mask[:, col])[0]
                if len(masked_indices) > 0:
                    sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                    sorted_image[masked_indices, col] = image[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            logger.debug("Sorting horizontally.")
            for row in range(image.shape[0]):
                masked_indices = np.where(pixel_mask[row, :])[0]
                if len(masked_indices) > 0:
                    sorted_indices = np.argsort(channel_to_sort[row, masked_indices])
                    sorted_image[row, masked_indices] = image[row, masked_indices[sorted_indices]]

        return sorted_image

    # Resize for efficient processing
    h, w = frame.shape[:2]
    resized_w = max(1, w // resolution)
    resized_h = max(1, h // resolution)
    resized_frame = cv2.resize(frame, (resized_w, resized_h))
    logger.debug(f"Resized frame to {resized_w}x{resized_h} for processing.")

    # Resize the mask to match the downscaled frame if provided
    if mask is not None:
        resized_mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        resized_mask = resized_mask.astype(bool)
        logger.debug(f"Resized mask to {resized_w}x{resized_h}.")
    else:
        resized_mask = np.ones(resized_frame.shape[:2], dtype=bool)  # Default mask is all True
        logger.debug("No mask provided. Using default mask (all True).")

    # Apply pixel sorting within the mask
    sorted_image = pixel_sort(
        resized_frame,
        resized_mask,
        sorting_direction,
        sorting_style,
        thresh_min,
        thresh_max
    )

    # Upscale back to original resolution
    output_frame = cv2.resize(sorted_image, (w, h))
    logger.debug(f"Upscaled sorted image back to original size {w}x{h}.")

    # Adjust contrast
    if contrast != 1.0:
        f = contrast
        output_frame = cv2.convertScaleAbs(output_frame, alpha=f, beta=0)
        logger.debug(f"Adjusted contrast by a factor of {f}.")

    # Adjust saturation
    if saturation != 1.0:
        hsv = cv2.cvtColor(output_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        output_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        logger.debug(f"Adjusted saturation by a factor of {saturation}.")

    # Add noise if required
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale, output_frame.shape).astype(np.uint8)
        output_frame = cv2.addWeighted(output_frame, 1 - (noise_amount / 100), noise, noise_amount / 100, 0)
        logger.debug(f"Added noise: Amount={noise_amount}%, Scale={noise_scale}.")

    # Blend with original frame based on opacity
    if opacity < 1.0:
        output_frame = cv2.addWeighted(frame, 1 - opacity, output_frame, opacity, 0)
        logger.debug(f"Blended sorted image with original frame at opacity {opacity}.")

    return output_frame
