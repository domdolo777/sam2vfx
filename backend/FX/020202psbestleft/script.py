import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_param(params, name, default, param_type, min_val=None, max_val=None):
    """Get and validate parameter values"""
    value = params.get(name, default)
    try:
        value = param_type(value)
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {name}. Using default: {default}")
        value = default
    return value

def rotate_image(image, angle, center=None):
    """Rotate image by given angle"""
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def pixel_sort(image, mask, axis, style, thresh_min, thresh_max):
    """Sort pixels based on specified criteria within mask"""
    # Convert image to desired color space
    if style == 0:  # Sort by Hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 0].astype(np.float32)
    elif style == 1:  # Sort by Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 1].astype(np.float32)
    else:  # Sort by Lightness
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channel_to_sort = grayscale.astype(np.float32)

    # Normalize the channel
    channel_to_sort = cv2.normalize(channel_to_sort, None, 0, 255, cv2.NORM_MINMAX)

    # Create threshold mask
    threshold_mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max)
    pixel_mask = threshold_mask & mask

    if not np.any(pixel_mask):
        logger.info("No pixels meet threshold and mask criteria")
        return image

    sorted_image = image.copy()

    # Sort pixels based on direction
    if axis == 0:  # Vertical sorting
        for col in range(image.shape[1]):
            masked_indices = np.where(pixel_mask[:, col])[0]
            if len(masked_indices) > 0:
                sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                sorted_image[masked_indices, col] = image[masked_indices[sorted_indices], col]
    else:  # Horizontal sorting
        for row in range(image.shape[0]):
            masked_indices = np.where(pixel_mask[row, :])[0]
            if len(masked_indices) > 0:
                sorted_indices = np.argsort(channel_to_sort[row, masked_indices])
                sorted_image[row, masked_indices] = image[row, masked_indices[sorted_indices]]

    return sorted_image

def apply_effect(frame, mask=None, **params):
    """Apply pixel sorting effect with parameters"""
    try:
        logger.info(f"Applying pixel sort effect with params: {params}")

        # Extract parameters with validation
        threshold_min = get_param(params, "threshold_min", 0, int, 0, 255)
        threshold_max = get_param(params, "threshold_max", 255, int, 0, 255)
        angle = get_param(params, "angle", 0, float, 0, 360)
        contrast = get_param(params, "contrast", 1.0, float, 0.5, 2.0)
        saturation = get_param(params, "saturation", 1.0, float, 0.5, 2.0)
        resolution = get_param(params, "resolution", 1, int, 1, 10)
        noise_amount = get_param(params, "noise_amount", 0, int, 0, 100)
        noise_scale = get_param(params, "noise_scale", 1.0, float, 0.1, 10.0)
        sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)
        sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)

        # Default mask if none provided
        if mask is None:
            mask = np.ones(frame.shape[:2], dtype=bool)

        # Apply rotation if needed
        if angle != 0:
            frame = rotate_image(frame, angle)

        # Resize for faster processing
        h, w = frame.shape[:2]
        resized_frame = cv2.resize(frame, (w // resolution, h // resolution))
        mask_resized = cv2.resize(mask, (w // resolution, h // resolution), 
                                interpolation=cv2.INTER_NEAREST)
        mask_resized = mask_resized.astype(bool)

        # Apply pixel sorting
        sorted_image = pixel_sort(resized_frame, mask_resized, sorting_direction, 
                                sorting_style, threshold_min, threshold_max)

        # Upscale back to original resolution
        output_frame = cv2.resize(sorted_image, (w, h))

        # Add noise if required
        if noise_amount > 0:
            noise = np.random.normal(0, noise_scale, output_frame.shape).astype(np.uint8)
            output_frame = cv2.addWeighted(output_frame, 1 - (noise_amount / 100), 
                                         noise, noise_amount / 100, 0)

        logger.info("Pixel sort effect applied successfully")
        return output_frame

    except Exception as e:
        logger.error(f"Error in pixel sort effect: {str(e)}")
        return frame