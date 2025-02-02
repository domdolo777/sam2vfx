import cv2
import numpy as np

# Alternative approach to pixel sorting to achieve the waterfall-like effect

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

def apply_alternative_pixel_sort(frame, **params):
    if not params:
        params = {}

    # Get parameters with defaults and ranges
    thresh_min = get_param(params, "threshold_min", 50, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 200, int, 0, 255)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 2, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness

    # Apply sorting logic based on lightness threshold
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
        if axis == 0:  # Vertical sorting
            for col in range(image.shape[1]):
                masked_indices = np.where(mask[:, col])[0]
                if len(masked_indices) > 1:
                    sorted_indices = np.argsort(channel_to_sort[masked_indices, col])
                    sorted_image[masked_indices, col] = image[masked_indices[sorted_indices], col]
        else:  # Horizontal sorting
            for row in range(image.shape[0]):
                masked_indices = np.where(mask[row, :])[0]
                if len(masked_indices) > 1:
                    sorted_indices = np.argsort(channel_to_sort[row, masked_indices])
                    sorted_image[row, masked_indices] = image[row, masked_indices[sorted_indices]]

        return sorted_image

    # Apply pixel sorting with no resizing for more precise results
    sorted_image = pixel_sort(frame, sorting_direction, sorting_style, thresh_min, thresh_max)

    return sorted_image

# Load the image
input_image_path = '/mnt/data/Screenshot 2024-10-09 at 9.34.57 AM.png'  # Replace with your image path
frame = cv2.imread(input_image_path)

# Apply the alternative pixel sorting effect
params = {
    "threshold_min": 50,
    "threshold_max": 200,
    "sorting_direction": 0,  # Vertical sorting
    "sorting_style": 2       # Sort by Lightness
}
output_frame = apply_alternative_pixel_sort(frame, **params)

# Save and display the result
output_image_path = '/mnt/data/pixel_sorted_alternative_output.png'
cv2.imwrite(output_image_path, output_frame)

import IPython.display as display
from PIL import Image

# Display the resulting image
output_image = Image.open(output_image_path)
display.display(output_image)
