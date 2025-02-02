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
def apply_effect(frame, **params):
    if not params:
        params = {}

    # Extract and validate parameters
    thresh_min = get_param(params, "threshold_min", 0.25, float, 0.0, 1.0)
    thresh_max = get_param(params, "threshold_max", 0.8, float, 0.0, 1.0)
    randomness = get_param(params, "randomness", 0, float, 0.0, 1.0)  # Controls how much randomness to introduce
    angle = get_param(params, "angle", 0, float, 0.0, 360.0)
    strip_width = get_param(params, "strip_width", 5, int, 1, 100)
    strip_spacing = get_param(params, "strip_spacing", 0, int, 0, 50)
    pulling_distance = get_param(params, "pulling_distance", 1.0, float, 0.1, 10.0)
    char_length = get_param(params, "char_length", 10, int, 1, 100)  # Characteristic length for interval width
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness

    # Convert frame to float for safe pixel manipulation
    original_dtype = frame.dtype
    if original_dtype != np.float32:
        frame = frame.astype(np.float32) / 255.0

    # Rotate image for angled pixel sorting
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    # Core pixel sorting logic
    def pixel_sort(image, style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance):
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0]
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1]
        else:  # Sort by Lightness
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale

        # Mask for threshold range
        mask = (channel_to_sort >= thresh_min * 255) & (channel_to_sort <= thresh_max * 255)
        sorted_image = image.copy()

        # Sort pixels based on thresholds
        for col in range(0, image.shape[1], strip_width + strip_spacing):
            masked_indices = np.where(mask[:, col])[0]
            if len(masked_indices) > 1:
                # Pull indices across pulling distance and sort
                pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[0] - 1).astype(int)
                sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                sorted_image[pulling_indices, col] = image[masked_indices[sorted_indices], col]

        return sorted_image

    # Apply pixel sorting with parameters
    rotated_frame = rotate_image(frame, angle)
    sorted_image = pixel_sort(rotated_frame, sorting_style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance)
    
    # Optionally introduce randomness by not sorting some intervals
    if randomness > 0:
        random_mask = np.random.rand(*sorted_image.shape[:2]) > randomness
        frame[random_mask] = sorted_image[random_mask]

    # Handle pixel format conversion before returning the frame
    result = np.clip(sorted_image, 0, 1)
    if original_dtype != np.float32:
        result = (result * 255).astype(np.uint8)
    
    return result
