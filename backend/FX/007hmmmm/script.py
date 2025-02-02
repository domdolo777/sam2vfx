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

def pixel_sort(image, mask, axis, style, thresh_min, thresh_max):
    # Convert image to the desired color space
    if style == 0:  # Sort by Hue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 0].astype(np.float32)  # Hue
    elif style == 1:  # Sort by Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_to_sort = hsv[:, :, 1].astype(np.float32)  # Saturation
    else:  # Sort by Lightness (Brightness)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channel_to_sort = grayscale.astype(np.float32)

    # Normalize the channel to ensure values are in the range [0, 255]
    channel_to_sort = cv2.normalize(channel_to_sort, None, 0, 255, cv2.NORM_MINMAX)

    # Create a threshold mask to control which pixels are sorted
    threshold_mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max)
    pixel_mask = threshold_mask & mask  # Combine with the input mask

    # Collect all pixels within the mask for sorting
    pixels_to_sort = np.where(pixel_mask)
    sorted_image = image.copy()

    if not np.any(pixel_mask):
        print("No pixels are within the specified threshold and mask.")
        return sorted_image  # Return original image if no pixels pass the mask

    # Sort pixels based on the selected direction (0 for vertical, 1 for horizontal)
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

def apply_effect(frame, mask=None, **params):
    if not params:
        params = {}

    # Extract parameters with validation
    threshold_min = get_param(params, "threshold_min", 0, int, 0, 255)
    threshold_max = get_param(params, "threshold_max", 255, int, 0, 255)
    angle = get_param(params, "angle", 0, float, 0, 360)
    resolution = get_param(params, "resolution", 1, int, 1, 10)
    noise_amount = get_param(params, "noise_amount", 0, int, 0, 100)
    noise_scale = get_param(params, "noise_scale", 1.0, float, 0.1, 10.0)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)
    sorting_style = get_param(params, "sorting_style", 0, int, 0, 2)
    opacity = get_param(params, "opacity", 1.0, float, 0.0, 1.0)
    expand = get_param(params, "expand", 0, int, -50, 50)  # Handle mask expansion

    # Apply rotation if needed
    if angle != 0:
        frame = rotate_image(frame, angle)

    # Resize for faster processing
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // resolution, h // resolution))

    # Handle mask resizing, using it only to pick pixels
    if mask is not None:
        mask = cv2.resize(mask, (w // resolution, h // resolution), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)

        # Apply mask expansion if needed
        if expand != 0:
            mask = cv2.dilate(mask.astype(np.uint8), None, iterations=expand) if expand > 0 else \
                   cv2.erode(mask.astype(np.uint8), None, iterations=abs(expand))
        mask = mask.astype(bool)

    else:
        mask = np.ones(resized_frame.shape[:2], dtype=bool)  # Use a default mask if none is provided

    # Apply pixel sorting, pixels selected by mask but the effect applies to the whole frame
    sorted_image = pixel_sort(resized_frame, mask, sorting_direction, sorting_style, threshold_min, threshold_max)

    # Upscale the sorted image back to the original resolution
    output_frame = cv2.resize(sorted_image, (w, h))

    # Add noise to the pixel-sorted areas only
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale * 255, output_frame.shape).astype(np.float32)
        noise = cv2.normalize(noise, None, -127, 127, cv2.NORM_MINMAX)
        noisy_area_mask = cv2.resize(mask.astype(np.uint8), (w, h)).astype(bool)

        # Blend noise only into the sorted regions
        output_frame[noisy_area_mask] = cv2.add(output_frame[noisy_area_mask], (noise[noisy_area_mask] * (noise_amount / 100)))
        output_frame = np.clip(output_frame, 0, 255).astype(np.uint8)

    # Blend with original frame based on opacity
    if opacity < 1.0:
        output_frame = cv2.addWeighted(frame, 1 - opacity, output_frame, opacity, 0)

    return output_frame
