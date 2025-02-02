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

def rotate_image(image, angle):
    # Get image dimensions
    h, w = image.shape[:2]
    # Get the center of the image
    center = (w / 2, h / 2)
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Adjust the rotation matrix to prevent cropping or misalignment
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    
    # Compute the new bounding dimensions of the image
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to account for the new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
    
    # Resize back to the original size to maintain consistency
    final_image = cv2.resize(rotated_image, (w, h))
    return final_image

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

    # Apply rotation based on the angle parameter
    if angle != 0:
        frame = rotate_image(frame, angle)

    # Pixel sorting function based on threshold values
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

        # Mask for pixel range
        mask = (channel_to_sort >= thresh_min) & (channel_to_sort <= thresh_max)

        # Sort pixels along specified axis while preserving shape
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

    # Resize for processing
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (w // resolution, h // resolution))

    # Apply pixel sorting
    sorted_image = pixel_sort(resized_frame, sorting_direction, sorting_style, thresh_min, thresh_max)

    # Add noise if specified
    if noise_amount > 0:
        noise = np.random.normal(0, noise_scale, sorted_image.shape).astype(np.uint8)
        sorted_image = cv2.addWeighted(sorted_image, 1 - (noise_amount / 100), noise, noise_amount / 100, 0)

    # Resize back to original resolution
    output_frame = cv2.resize(sorted_image, (w, h))

    return output_frame
