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

# Function to apply pixel sorting
def apply_effect(frame, mask, **params):
    if not params:
        params = {}

    # Extract and validate parameters
    thresh_min = get_param(params, "threshold_min", 0.25, float, 0.0, 1.0)
    thresh_max = get_param(params, "threshold_max", 0.8, float, 0.0, 1.0)
    randomness = get_param(params, "randomness", 0, float, 0.0, 1.0)
    angle = get_param(params, "angle", 0, float, -360.0, 360.0)
    strip_width = get_param(params, "strip_width", 5, int, 1, 100)
    strip_spacing = get_param(params, "strip_spacing", 0, int, 0, 50)
    pulling_distance = get_param(params, "pulling_distance", 1.0, float, 0.1, 10.0)
    char_length = get_param(params, "char_length", 10, int, 1, 100)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness
    noise_strength = get_param(params, "noise_strength", 0.0, float, 0.0, 1.0)
    invert_mask = get_param(params, "invert_mask", False, bool)

    # Convert frame to float for safe pixel manipulation
    original_dtype = frame.dtype
    if original_dtype != np.float32:
        frame = frame.astype(np.float32) / 255.0

    # Invert mask if requested
    if invert_mask:
        mask = ~mask

    # Adjust strip direction (angle without rotating the entire image)
    def rotate_strips(image, angle):
        if angle != 0:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            return rotated
        return image

    # Core pixel sorting logic
    def pixel_sort(image, mask, style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance):
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0] / 255.0  # Normalize
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1] / 255.0  # Normalize
        else:  # Sort by Lightness (Grayscale)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale / 255.0  # Normalize the grayscale values to 0-1

        # Mask for threshold range (only pixels within mask region)
        threshold_mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max) & mask.astype(bool)

        sorted_image = image.copy()

        # Sort pixels based on thresholds and intervals
        for col in range(0, image.shape[1], strip_width + strip_spacing):
            masked_indices = np.where(threshold_mask[:, col])[0]
            if len(masked_indices) > 1:
                # Pull indices across pulling distance and sort
                pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[0] - 1).astype(int)
                sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                sorted_image[pulling_indices, col] = image[masked_indices[sorted_indices], col]

        return sorted_image

    # Rotate strip direction without affecting the whole frame
    frame_with_rotated_strips = rotate_strips(frame, angle)

    # Apply pixel sorting with parameters and mask
    sorted_image = pixel_sort(frame_with_rotated_strips, mask, sorting_style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance)
    
    # Optionally introduce randomness by not sorting some intervals
    if randomness > 0:
        random_mask = np.random.rand(*sorted_image.shape[:2]) > randomness
        frame[random_mask] = sorted_image[random_mask]

    # Resize output back to original dimensions
    if original_dtype != np.float32:
        sorted_image = (sorted_image * 255).astype(np.uint8)

    return sorted_image
