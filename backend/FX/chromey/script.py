import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_effect(frame: np.ndarray, mask: np.ndarray = None, shift_x: int = 5, 
                shift_y: int = 0, intensity: float = 1.0) -> np.ndarray:
    """
    Apply chromatic aberration effect to the frame by shifting color channels.

    Args:
        frame: Input frame (BGR format)
        mask: Mask where effect should be applied
        shift_x: Horizontal shift in pixels
        shift_y: Vertical shift in pixels
        intensity: Effect intensity (0.0 to 50.0)

    Returns:
        Frame with chromatic aberration applied
    """
    try:
        logger.info(f"Applying chromatic aberration with params: x={shift_x}, y={shift_y}, intensity={intensity}")
        
        if intensity <= 0.0:
            return frame.copy()

        if mask is None:
            mask = np.ones(frame.shape[:2], dtype=bool)
        
        # Split the channels
        b, g, r = cv2.split(frame)
        
        # Calculate shifts based on intensity
        shift_x_r = int(shift_x * (intensity / 50))
        shift_y_r = int(shift_y * (intensity / 50))
        
        shift_x_b = int(-shift_x * (intensity / 50))
        shift_y_b = int(-shift_y * (intensity / 50))
        
        # Create translation matrices
        M_r = np.float32([[1, 0, shift_x_r], [0, 1, shift_y_r]])
        M_b = np.float32([[1, 0, shift_x_b], [0, 1, shift_y_b]])
        
        # Apply shifts only to masked area
        r_shifted = cv2.warpAffine(r, M_r, (frame.shape[1], frame.shape[0]))
        b_shifted = cv2.warpAffine(b, M_b, (frame.shape[1], frame.shape[0]))
        
        # Merge channels back
        aberrated_frame = cv2.merge([b_shifted, g, r_shifted])
        
        # Apply mask
        mask_3d = np.stack([mask] * 3, axis=2)
        result = np.where(mask_3d, aberrated_frame, frame)
        
        logger.info("Chromatic aberration effect applied successfully")
        return result

    except Exception as e:
        logger.error(f"Error in chromatic aberration effect: {str(e)}")
        return frame
