import cv2
import numpy as np

def apply_effect(frame: np.ndarray, dot_size: int = 10, light_threshold: float = 0.5) -> np.ndarray:
    """Applies a color halftone effect triggered by light values, using bicubic sampling.
    Args:
        frame (np.ndarray): The input frame as a NumPy array.
        dot_size (int, optional): Size of halftone dots. Defaults to 10.
        light_threshold (float, optional): Luminance threshold (0.0-1.0). Defaults to 0.5.
    Returns:
        np.ndarray: The processed frame with the halftone effect.
    """

    original_dtype = frame.dtype
    frame = frame.astype(np.float32) / 255.0

    b, g, r = cv2.split(frame)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    h, w = frame.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)

    def create_halftone(channel, angle):
        x_rot = X * np.cos(angle) - Y * np.sin(angle)
        y_rot = X * np.sin(angle) + Y * np.cos(angle)

        x_scaled = x_rot / dot_size
        y_scaled = y_rot / dot_size

        pattern = np.sqrt((x_scaled % 1 - 0.5)**2 + (y_scaled % 1 - 0.5)**2)

        channel_resized = cv2.resize(channel, (w // dot_size, h // dot_size), interpolation=cv2.INTER_CUBIC)
        channel_upscaled = cv2.resize(channel_resized, (w, h), interpolation=cv2.INTER_CUBIC)
        halftone = (channel_upscaled > pattern).astype(np.float32)
        return halftone

    b_half = create_halftone(b, np.pi/6)
    g_half = create_halftone(g, np.pi/3)
    r_half = create_halftone(r, 0)

    result = np.zeros_like(frame)
    result[:,:,0] = np.where(luminance > light_threshold, b_half, b)
    result[:,:,1] = np.where(luminance > light_threshold, g_half, g)
    result[:,:,2] = np.where(luminance > light_threshold, r_half, r)

    result = np.clip(result, 0, 1)
    if original_dtype != np.float32:
        result = (result * 255).astype(original_dtype)

    return result