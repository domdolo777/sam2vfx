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

    # Placeholder segmentation mask (replace with actual mask from your process)
    segmentation_mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Assuming a grayscale mask
    segmentation_mask[100:200, 150:300] = 255  # Example rectangle mask area

    # Apply pixel sorting only on the masked area
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

        # Apply the mask to isolate the pixels within the segmentation
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # Create a blank output image
        sorted_image = masked_image.copy()

        if axis == 0:  # Vertical sorting
            for col in range(masked_image.shape[1]):
                masked_indices = np.where(mask[:, col] == 255)[0]
                if len(masked_indices) > 0:
                    # Extract only masked pixels to sort
                    values_to_sort = channel_to_sort[masked_indices, col]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[masked_indices, col] = masked_image[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            for row in range(masked_image.shape[0]):
                masked_indices = np.where(mask[row, :] == 255)[0]
                if len(masked_indices) > 0:
                    # Extract only masked pixels to sort
                    values_to_sort = channel_to_sort[row, masked_indices]
                    sorted_indices = np.argsort(values_to_sort)
                    sorted_image[row, masked_indices] = masked_image[row, masked_indices[sorted_indices]]

        return sorted_image

    # Isolate the region and apply pixel sorting
    sorted_image = pixel_sort(frame, segmentation_mask, sorting_direction, sorting_style, thresh_min, thresh_max)

    # Combine the original image and the sorted image using the mask
    final_image = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(segmentation_mask))
    final_image = cv2.add(final_image, cv2.bitwise_and(sorted_image, sorted_image, mask=segmentation_mask))

    return final_image
