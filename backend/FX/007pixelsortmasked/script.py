import cv2
import numpy as np

def get_param(params, name, default, param_type, min_val=None, max_val=None):
    value = params.get(name, default)
    try:
        value = param_type(value)
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
    except (ValueError, TypeError):
        value = default
    return value

def apply_effect(frame, mask=None, **params):
    if not params:
        params = {}

    # Extract parameters with validation
    thresh_min = get_param(params, "threshold_min", 0, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 255, int, 0, 255)
    contrast = get_param(params, "contrast", 1.0, float, 0.5, 2.0)
    saturation = get_param(params, "saturation", 1.0, float, 0.5, 2.0)
    resolution = get_param(params, "resolution", 1, int, 1, 10)
    noise_amount = get_param(params, "noise_amount", 0, int, 0, 100)
    noise_scale = get_param(params, "noise_scale", 1.0, float, 0.1, 10.0)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)

    # Ensure a mask is provided
    if mask is not None:
        mask = mask.astype(bool)  # Convert mask to boolean array
        masked_frame = np.zeros_like(frame)  # Create an empty frame for the mask area
        masked_frame[mask] = frame[mask]  # Only process the masked area

        # Apply the pixel sorting effect only within the mask
        sorted_frame = pixel_sort(masked_frame, mask, sorting_direction, sorting_style, thresh_min, thresh_max)

        # Blend the sorted pixels back into the original frame using the mask
        frame[mask] = sorted_frame[mask]

    return frame

def pixel_sort(image, mask, axis, style, thresh_min, thresh_max):
    # Example sorting logic based on brightness for demonstration
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the mask to limit the effect to the segmented area
    if mask is not None:
        mask = mask.astype(bool)  # Ensure the mask is a boolean array
        grayscale = np.where(mask, grayscale, 0)

    # Create a mask for pixels within the threshold range
    mask_range = (grayscale >= thresh_min) & (grayscale <= thresh_max)

    # Apply sorting logic based on the axis (0 for vertical, 1 for horizontal)
    sorted_image = image.copy()
    if axis == 0:  # Vertical sorting
        for col in range(image.shape[1]):
            masked_indices = np.where(mask_range[:, col])[0]
            if len(masked_indices) > 0:
                values_to_sort = grayscale[masked_indices, col]
                sorted_indices = np.argsort(values_to_sort)
                sorted_image[masked_indices, col] = image[masked_indices[sorted_indices], col]
    else:  # Horizontal sorting
        for row in range(image.shape[0]):
            masked_indices = np.where(mask_range[row, :])[0]
            if len(masked_indices) > 0:
                values_to_sort = grayscale[row, masked_indices]
                sorted_indices = np.argsort(values_to_sort)
                sorted_image[row, masked_indices] = image[row, masked_indices[sorted_indices]]

    return sorted_image
