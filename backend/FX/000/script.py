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
    angle = get_param(params, "angle", 0, float, 0, 360)
    contrast = get_param(params, "contrast", 1.0, float, 0.5, 2.0)
    saturation = get_param(params, "saturation", 1.0, float, 0.5, 2.0)
    resolution = get_param(params, "resolution", 1, int, 1, 10)
    noise_amount = get_param(params, "noise_amount", 0, int, 0, 100)
    noise_scale = get_param(params, "noise_scale", 1, float, 0.1, 10.0)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness

    # Apply sorting logic based on threshold
    def pixel_sort(image, axis, style, thresh_min, thresh_max):
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
        if axis == 0:
            for col in range(image.shape[1]):
                masked_indices = np.where(mask[:, col])[0]
                if len(masked_indices) > 0:
                    sorted_indices = masked_indices[np.argsort(channel_to_sort[masked_indices, col])]
                    sorted_image[masked_indices, col] = image[sorted_indices, col]
        else:
            for row in range(image.shape[0]):
                masked_indices = np.where(mask[row, :])[0]
                if len(masked_indices) > 0:
                    sorted_indices = masked_indices[np.argsort(channel_to_sort[row, masked_indices])]
                    sorted_image[row, masked_indices] = image[row, sorted_indices]

        return sorted_image

    # Resize the image based on resolution
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // resolution, h // resolution))

    # Apply pixel sorting
    sorted_image = pixel_sort(resized_frame, sorting_direction, sorting_style, thresh_min, thresh_max)

    # Apply noise if needed
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale, sorted_image.shape).astype(np.uint8)
        sorted_image = cv2.addWeighted(sorted_image, 1 - (noise_amount / 100), noise, noise_amount / 100, 0)

    # Resize back to original resolution
    output_frame = cv2.resize(sorted_image, (w, h))

    return output_frame