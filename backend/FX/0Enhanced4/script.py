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

def apply_effect(frame, sam2_mask=None, **params):
    if not params:
        params = {}

    # Check if sam2_mask is provided
    if sam2_mask is None:
        raise ValueError("SAM2 mask must be provided to apply the effect.")

    # Ensure the SAM2 mask is resized and binary
    h, w = frame.shape[:2]
    if sam2_mask.shape != (h, w):
        sam2_mask = cv2.resize(sam2_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    sam2_mask = cv2.threshold(sam2_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Extract parameters with defaults and value clamping
    thresh_min = get_param(params, "threshold_min", 0, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 255, int, 0, 255)
    angle = get_param(params, "angle", 0, float, 0, 360)
    contrast = get_param(params, "contrast", 1.0, float, 0.5, 2.0)
    saturation = get_param(params, "saturation", 1.0, float, 0.5, 2.0)
    resolution = get_param(params, "resolution", 1, int, 1, 10)
    noise_amount = get_param(params, "noise_amount", 0, int, 0, 100)
    noise_scale = get_param(params, "noise_scale", 1.0, float, 0.1, 10.0)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness

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

        # Apply mask to restrict sorting to the masked area
        sorted_image = image.copy()

        if axis == 0:  # Vertical sorting
            for col in range(mask.shape[1]):
                masked_indices = np.where(mask[:, col] == 255)[0]
                if len(masked_indices) > 0:
                    values_to_sort = channel_to_sort[masked_indices, col]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[masked_indices, col] = image[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            for row in range(mask.shape[0]):
                masked_indices = np.where(mask[row, :] == 255)[0]
                if len(masked_indices) > 0:
                    values_to_sort = channel_to_sort[row, masked_indices]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[row, masked_indices] = image[row, masked_indices[sorted_indices]]

        return sorted_image

    # Downscale for processing efficiency
    resized_frame = cv2.resize(frame, (w // resolution, h // resolution))
    resized_mask = cv2.resize(sam2_mask, (w // resolution, h // resolution), interpolation=cv2.INTER_NEAREST)
    sorted_image = pixel_sort(resized_frame, resized_mask, sorting_direction, sorting_style, thresh_min, thresh_max)

    # Upscale back to original resolution
    output_frame = cv2.resize(sorted_image, (w, h))

    # Add noise if required
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale, output_frame.shape).astype(np.uint8)
        output_frame = cv2.addWeighted(output_frame, 1 - (noise_amount / 100), noise, noise_amount / 100, 0)

    # Combine original and processed images using the mask
    final_image = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(sam2_mask))
    final_image = cv2.add(final_image, cv2.bitwise_and(output_frame, output_frame, mask=sam2_mask))

    return final_image
