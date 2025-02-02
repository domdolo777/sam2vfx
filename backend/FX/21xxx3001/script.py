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

# Function to apply noise to the image
def add_noise(image, noise_factor):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

# Function to apply pixel sorting
def pixel_sort(image, mask, direction, style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance):
    # Channel selection based on sorting style
    if style == 0:  # Sort by Hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 0]
    elif style == 1:  # Sort by Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 1]
    else:  # Sort by Lightness (Luminance in Lab color space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        channel_to_sort = lab[:, :, 0]  # L channel (lightness)

    # Apply thresholds (convert to 0-255 range)
    min_thresh_value = int(thresh_min * 255)
    max_thresh_value = int(thresh_max * 255)

    # Threshold mask for sorting
    threshold_mask = (channel_to_sort >= min_thresh_value) & (channel_to_sort <= max_thresh_value)

    sorted_image = image.copy()

    # Depending on the direction (horizontal or vertical) apply the sorting
    if direction == "horizontal":
        # Sort along rows (horizontal direction)
        for row in range(0, image.shape[0], strip_width + strip_spacing):
            masked_indices = np.where(threshold_mask[row, :])[0]
            if len(masked_indices) > 1:
                # Pull and sort pixels horizontally (left or right)
                pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[1] - 1).astype(int)
                sorted_indices = np.argsort(channel_to_sort[row, masked_indices])
                sorted_image[row, pulling_indices] = image[row, masked_indices[sorted_indices]]
    else:
        # Sort along columns (vertical direction)
        for col in range(0, image.shape[1], strip_width + strip_spacing):
            masked_indices = np.where(threshold_mask[:, col])[0]
            if len(masked_indices) > 1:
                # Pull and sort pixels vertically (up or down)
                pulling_indices = np.clip(masked_indices * pulling_distance, 0, image.shape[0] - 1).astype(int)
                sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                sorted_image[pulling_indices, col] = image[masked_indices[sorted_indices], col]

    return sorted_image

# Core function to apply pixel sorting with directional control
def apply_effect(frame, mask=None, **params):
    if not params:
        params = {}

    # Extract and validate parameters
    thresh_min = get_param(params, "threshold_min", 0.0, float, 0.0, 1.0)
    thresh_max = get_param(params, "threshold_max", 0.8, float, 0.0, 1.0)
    randomness = get_param(params, "randomness", 0, float, 0.0, 1.0)
    direction = get_param(params, "direction", 0, int, 0, 1)  # 0 = Horizontal, 1 = Vertical
    strip_width = get_param(params, "strip_width", 5, int, 1, 100)
    strip_spacing = get_param(params, "strip_spacing", 1, int, -50, 50)
    pulling_distance = get_param(params, "pulling_distance", 1.0, float, 0.1, 10.0)
    char_length = get_param(params, "char_length", 10, int, 1, 100)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)
    noise_factor = get_param(params, "noise_strength", 0.0, float, 0.0, 1.0)

    # Convert direction from integer to string for sorting
    if direction == 0:
        direction_str = "horizontal"
    else:
        direction_str = "vertical"

    # Convert frame to float for safe pixel manipulation
    original_dtype = frame.dtype
    if original_dtype != np.float32:
        frame = frame.astype(np.float32) / 255.0

    # Apply noise if required
    if noise_factor > 0:
        frame = add_noise(frame, noise_factor)

    # Apply pixel sorting based on the specified direction
    sorted_image = pixel_sort(frame, mask, direction_str, sorting_style, thresh_min, thresh_max, strip_width, strip_spacing, pulling_distance)

    # Optionally introduce randomness by not sorting some intervals
    if randomness > 0:
        random_mask = np.random.rand(*sorted_image.shape[:2]) > randomness
        frame[random_mask] = sorted_image[random_mask]

    # Resize output back to original dimensions
    if original_dtype != np.float32:
        sorted_image = (sorted_image * 255).astype(np.uint8)

    return sorted_image
