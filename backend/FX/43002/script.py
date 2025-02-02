import cv2
import numpy as np

# Helper function for parameter extraction with validation
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

# Updated function to handle sorting in both directions and handle the mask correctly
def apply_effect(frame, mask, **params):
    if not params:
        params = {}

    # Extract and validate parameters
    thresh_min = get_param(params, "threshold_min", 0.25, float, 0.0, 1.0)
    thresh_max = get_param(params, "threshold_max", 0.8, float, 0.0, 1.0)
    randomness = get_param(params, "randomness", 0, float, 0.0, 1.0)
    direction = get_param(params, "direction", "horizontal", str)  # Use "horizontal" or "vertical"
    strip_width = get_param(params, "strip_width", 5, int, 1, 100)
    strip_spacing = get_param(params, "strip_spacing", 1, int, -50, 50)
    pulling_distance = get_param(params, "pulling_distance", 1.0, float, -10.0, 10.0)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness
    invert_mask = get_param(params, "invert_mask", False, bool)  # Apply sorting to inverted mask area

    # Convert frame to float for safe pixel manipulation
    original_dtype = frame.dtype
    if original_dtype != np.float32:
        frame = frame.astype(np.float32) / 255.0

    # Core pixel sorting logic
    def pixel_sort(image, mask, style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance):
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0]
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1]
        else:  # Sort by Lightness
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale

        # Apply thresholds (convert to 0-255 range)
        min_thresh_value = int(thresh_min * 255)
        max_thresh_value = int(thresh_max * 255)

        # Create threshold mask for sorting
        threshold_mask = (channel_to_sort >= min_thresh_value) & (channel_to_sort <= max_thresh_value)
        
        # Handle the mask inversion properly
        if invert_mask:
            mask = np.logical_not(mask)  # Invert the mask if the "invert_mask" is True
        
        # Combine mask and threshold mask
        combined_mask = threshold_mask & mask.astype(bool)

        sorted_image = image.copy()

        # Sort pixels based on direction
        if direction == "horizontal":
            for col in range(0, image.shape[1], strip_width + strip_spacing):
                masked_indices = np.where(combined_mask[:, col])[0]
                if len(masked_indices) > 1:
                    pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[0] - 1).astype(int)
                    sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                    sorted_image[pulling_indices, col] = image[masked_indices[sorted_indices], col]
        else:  # Vertical sorting
            for row in range(0, image.shape[0], strip_width + strip_spacing):
                masked_indices = np.where(combined_mask[row, :])[0]
                if len(masked_indices) > 1:
                    pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[1] - 1).astype(int)
                    sorted_indices = np.argsort(channel_to_sort[row, masked_indices])
                    sorted_image[row, pulling_indices] = image[row, masked_indices[sorted_indices]]

        return sorted_image

    # Apply pixel sorting with parameters and mask
    sorted_image = pixel_sort(frame, mask, sorting_style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance)

    # Optionally introduce randomness by not sorting some intervals
    if randomness > 0:
        random_mask = np.random.rand(*sorted_image.shape[:2]) > randomness
        frame[random_mask] = sorted_image[random_mask]

    # Resize output back to original dimensions
    if original_dtype != np.float32:
        sorted_image = (sorted_image * 255).astype(np.uint8)

    return sorted_image
