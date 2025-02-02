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

def rotate_image(image, angle, center=None):
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def apply_effect(frame, mask=None, **params):
    if not params:
        params = {}

    # Extract parameters with validation - ensure consistency with config.json
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

    # Apply rotation to the frame if needed
    if angle != 0:
        frame = rotate_image(frame, angle)

    def pixel_sort(image, mask, axis, style, thresh_min, thresh_max):
        if style == 0:  # Sort by Hue
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 0]
        elif style == 1:  # Sort by Saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            channel_to_sort = hsv[:, :, 1]
        else:  # Sort by Lightness (brightness)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channel_to_sort = grayscale

        # Combine pixel intensity thresholding and mask
        pixel_mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max) & mask

        # Ensure the mask is strictly respected
        sorted_image = image.copy()
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

    # Resize for efficient processing
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // resolution, h // resolution))

    # Resize the mask to match the downscaled frame if provided
    if mask is not None:
        mask = cv2.resize(mask, (w // resolution, h // resolution), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
    else:
        mask = np.ones(resized_frame.shape[:2], dtype=bool)

    # Apply pixel sorting within the mask
    sorted_image = pixel_sort(resized_frame, mask, sorting_direction, sorting_style, threshold_min, threshold_max)

    # Upscale back to original resolution
    output_frame = cv2.resize(sorted_image, (w, h))

    # Add noise if required
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale, output_frame.shape).astype(np.uint8)
        output_frame = cv2.addWeighted(output_frame, 1 - (noise_amount / 100), noise, noise_amount / 100, 0)

    return output_frame
