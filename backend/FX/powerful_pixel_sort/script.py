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
    Applies a powerful pixel sorting effect with advanced customization options.

    Parameters:
        frame (np.ndarray): The input video frame in BGR format.
        threshold_min (int): Minimum pixel brightness for sorting.
        threshold_max (int): Maximum pixel brightness for sorting.
        sorting_direction (int): Sorting direction (0: vertical, 1: horizontal).
        sorting_style (int): Sorting style (0: Hue, 1: Saturation, 2: Lightness).
        sort_in (bool): Enables pixel sorting from top to bottom or left to right.
        sort_out (bool): Enables pixel sorting from bottom to top or right to left.
        sort_size (int): Defines the size of the sorting window.
        intensity (float): Intensity of the pixel sorting effect.

    Returns:
        np.ndarray: The frame with the powerful pixel sorting effect applied.
    """
    if not params:
        params = {}

    # Get parameters with defaults and ranges
    thresh_min = get_param(params, "threshold_min", 50, int, 0, 255)
    thresh_max = get_param(params, "threshold_max", 200, int, 0, 255)
    sorting_direction = get_param(params, "sorting_direction", 0, int, 0, 1)  # 0: vertical, 1: horizontal
    sorting_style = get_param(params, "sorting_style", 2, int, 0, 2)  # 0: Hue, 1: Saturation, 2: Lightness
    sort_in = get_param(params, "sort_in", True, bool)
    sort_out = get_param(params, "sort_out", False, bool)
    sort_size = get_param(params, "sort_size", 5, int, 1, 20)
    intensity = get_param(params, "intensity", 1.0, float, 0.1, 10.0)

    # Validate parameters
    if thresh_min > thresh_max:
        thresh_min, thresh_max = thresh_max, thresh_min  # Swap to ensure thresh_min <= thresh_max

    # Apply sorting logic based on selected style
    def pixel_sort(image: np.ndarray, axis: int, style: int, thresh_min: int, thresh_max: int, sort_in: bool, sort_out: bool, sort_size: int, intensity: float) -> np.ndarray:
        """
        Sorts pixels in the image based on the specified channel and thresholds with advanced options.

        Args:
            image (np.ndarray): The input image.
            axis (int): Axis along which to sort (0: vertical, 1: horizontal).
            style (int): Channel to sort by (0: Hue, 1: Saturation, 2: Lightness).
            thresh_min (int): Minimum threshold for sorting.
            thresh_max (int): Maximum threshold for sorting.
            sort_in (bool): Enables sorting from start to end.
            sort_out (bool): Enables sorting from end to start.
            sort_size (int): Size of the sorting window.
            intensity (float): Intensity multiplier for sorting.

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

        sorted_image = image.copy()

        # Determine sorting order
        if sort_in and not sort_out:
            iterator = range(0, image.shape[axis])
        elif sort_out and not sort_in:
            iterator = range(image.shape[axis]-1, -1, -1)
        else:
            iterator = range(0, image.shape[axis])

        for i in iterator:
            if axis == 0:  # Vertical sorting
                window = slice(max(i - sort_size, 0), min(i + sort_size + 1, image.shape[0]))
                masked_indices = np.where(mask[window, :])[0]
                if len(masked_indices) > 1:
                    sorted_indices = np.argsort(channel_to_sort[window, masked_indices])
                    sorted_image[window, masked_indices] = image[window, masked_indices[sorted_indices]] * intensity
            else:  # Horizontal sorting
                window = slice(max(i - sort_size, 0), min(i + sort_size + 1, image.shape[1]))
                masked_indices = np.where(mask[:, window])[0]
                if len(masked_indices) > 1:
                    sorted_indices = np.argsort(channel_to_sort[masked_indices, window])
                    sorted_image[masked_indices, window] = image[masked_indices, window[sorted_indices]] * intensity

        return sorted_image

    # Apply pixel sorting with advanced options
    sorted_image = pixel_sort(
        frame,
        sorting_direction,
        sorting_style,
        thresh_min,
        thresh_max,
        sort_in,
        sort_out,
        sort_size,
        intensity
    )

    return sorted_image
