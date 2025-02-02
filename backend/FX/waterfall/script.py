import cv2
import numpy as np
from typing import Any, Dict

def get_param(params: Dict[str, Any], name: str, default: Any, param_type: type, min_val: Any = None, max_val: Any = None) -> Any:
    """
    Retrieves and validates a parameter from the params dictionary.

    Args:
        params (Dict[str, Any]): Dictionary of parameters.
        name (str): Name of the parameter to retrieve.
        default (Any): Default value if the parameter is not provided or invalid.
        param_type (type): Expected type of the parameter.
        min_val (Any, optional): Minimum allowable value. Defaults to None.
        max_val (Any, optional): Maximum allowable value. Defaults to None.

    Returns:
        Any: Validated parameter value.
    """
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

def apply_effect(frame: np.ndarray, **params) -> np.ndarray:
    """
    Applies a waterfall-like pixel sorting effect based on lightness.

    Args:
        frame (np.ndarray): The input video frame in BGR format.
        threshold_min (int, optional): Minimum pixel brightness for sorting. Defaults to 50.
        threshold_max (int, optional): Maximum pixel brightness for sorting. Defaults to 200.
        sorting_direction (int, optional): Sorting direction (0: vertical, 1: horizontal). Defaults to 0.
        sorting_style (int, optional): Sorting style (0: Hue, 1: Saturation, 2: Lightness). Defaults to 2.

    Returns:
        np.ndarray: The frame with the pixel sorting effect applied.
    """
    if not params:
        params = {}

    # Get parameters with defaults and ranges
    thresh_min = get_param(params, "threshold_min", 50, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 200, int, 0, 255)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 2, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness

    # Validate parameters
    if thresh_min > thresh_max:
        thresh_min, thresh_max = thresh_max, thresh_min  # Swap to ensure thresh_min <= thresh_max

    # Apply sorting logic based on selected style
    def pixel_sort(image: np.ndarray, axis: int, style: int, thresh_min: int, thresh_max: int) -> np.ndarray:
        """
        Sorts pixels in the image based on the specified channel and thresholds.

        Args:
            image (np.ndarray): The input image.
            axis (int): Axis along which to sort (0: vertical, 1: horizontal).
            style (int): Channel to sort by (0: Hue, 1: Saturation, 2: Lightness).
            thresh_min (int): Minimum threshold for sorting.
            thresh_max (int): Maximum threshold for sorting.

        Returns:
            np.ndarray: The image with sorted pixels.
        """
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
