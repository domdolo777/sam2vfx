import cv2
import numpy as np

def apply_effect(frame: np.ndarray, shift_x: int = 5, shift_y: int = 0, intensity: float = 1.0) -> np.ndarray:
    """
    Applies a chromatic aberration effect to the frame by shifting the red and blue channels.

    Args:
        frame (np.ndarray): The input frame image in BGR format.
        shift_x (int): Number of pixels to shift horizontally.
        shift_y (int): Number of pixels to shift vertically.
        intensity (float): Intensity of the effect (0.0 to 50.0).

    Returns:
        np.ndarray: The frame with chromatic aberration applied.
    """
    if intensity <= 0.0:
        return frame.copy()
    
    # Split the channels
    b, g, r = cv2.split(frame)
    
    # Calculate the shift amounts based on intensity (scale it within a reasonable range)
    shift_x_r = int(shift_x * (intensity / 50))  # Adjusting intensity scale to 50
    shift_y_r = int(shift_y * (intensity / 50))
    
    shift_x_b = int(-shift_x * (intensity / 50))
    shift_y_b = int(-shift_y * (intensity / 50))
    
    # Create translation matrices
    M_r = np.float32([[1, 0, shift_x_r], [0, 1, shift_y_r]])
    M_b = np.float32([[1, 0, shift_x_b], [0, 1, shift_y_b]])
    
    # Apply shifts
    r_shifted = cv2.warpAffine(r, M_r, (frame.shape[1], frame.shape[0]))
    b_shifted = cv2.warpAffine(b, M_b, (frame.shape[1], frame.shape[0]))
    
    # Merge channels back
    aberrated_frame = cv2.merge([b_shifted, g, r_shifted])
    
    return aberrated_frame

