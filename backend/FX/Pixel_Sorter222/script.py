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

def apply_effect(frame, **params):
    if not params:
        params = {}

    # Get parameters with defaults and ranges
    thresh_min = get_param(params, "threshold_min", 0, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 255, int, 0, 255)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness
    strip_width = get_param(params, "strip_width", 5, int, 1, 100)  # New: Controls width of each strip
    strip_spacing = get_param(params, "strip_spacing", 0, int, 0, 50)  # New: Spacing between strips
    pulling_distance = get_param(params, "pulling_distance", 1.0, float, 0.1, 10.0)  # New: Controls how far the strips pull

    # Apply sorting logic based on threshold
    def pixel_sort(image, axis, style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance):
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0]
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1]
        else:  # Sort by Lightness (brightness)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale

        # Create a mask for pixels within the threshold range
        mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max)

        # Sort pixels along the specified axis while preserving the original image shape
        sorted_image = image.copy()
        if axis == 0:  # Vertical sorting
            for col in range(0, image.shape[1], strip_width + strip_spacing):
                masked_indices = np.where(mask[:, col])[0]
                if len(masked_indices) > 1:
                    # Apply pulling distance (stretch the strip)
                    pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[0] - 1).astype(int)
                    sorted_indices = masked_indices[np.argsort(channel_to_sort[masked_indices, col])]
                    sorted_image[pulling_indices, col] = image[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            for row in range(0, image.shape[0], strip_width + strip_spacing):
                masked_indices = np.where(mask[row, :])[0]
                if len(masked_indices) > 1:
                    # Apply pulling distance (stretch the strip)
                    pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[1] - 1).astype(int)
                    sorted_indices = masked_indices[np.argsort(channel_to_sort[row, masked_indices])]
                    sorted_image[row, pulling_indices] = image[row, masked_indices]

        return sorted_image

    # Apply pixel sorting
    sorted_image = pixel_sort(frame, sorting_direction, sorting_style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance)

    return sorted_image
