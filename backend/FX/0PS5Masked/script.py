import cv2
import numpy as np

# Helper function to validate and retrieve parameter values
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

# Main effect application function
def apply_effect(frame, mask=None, **params):
    if not params:
        params = {}

    # Extract parameters with validation - ensure consistency with config.json
    threshold_min = get_param(params, "threshold_min", 0, int, 0, 255)
    threshold_max = get_param(params, "threshold_max", 255, int, 0, 255)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)

    # Ensure mask is provided and resize if necessary
    if mask is not None:
        mask = mask.astype(bool)
        
        # Resize the mask to match the frame size if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Apply pixel sorting effect only within the mask
        sorted_frame = pixel_sort(frame, mask, sorting_direction, sorting_style, threshold_min, threshold_max)

        # Blend the sorted pixels back into the original frame using the mask
        frame[mask] = sorted_frame[mask]

    return frame

# Pixel sorting function
def pixel_sort(image, mask, axis, style, threshold_min, threshold_max):
    # Create a mask for pixels within the threshold range based on brightness
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale = np.where(mask, grayscale, 0)  # Apply the mask to the grayscale version of the image
    mask_range = (grayscale >= threshold_min) & (grayscale <= threshold_max)

    # Prepare sorted image
    sorted_image = np.copy(image)

    # Apply sorting for each channel (BGR)
    for channel in range(3):  # Loop over B, G, R channels
        channel_data = image[:, :, channel]

        if axis == 0:  # Vertical sorting
            for col in range(image.shape[1]):
                masked_indices = np.where(mask_range[:, col])[0]
                if len(masked_indices) > 0:
                    values_to_sort = channel_data[masked_indices, col]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[masked_indices, col, channel] = channel_data[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            for row in range(image.shape[0]):
                masked_indices = np.where(mask_range[row, :])[0]
                if len(masked_indices) > 0:
                    values_to_sort = channel_data[row, masked_indices]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[row, masked_indices, channel] = channel_data[row, masked_indices[sorted_indices]]

    return sorted_image
