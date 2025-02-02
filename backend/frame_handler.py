import os
import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image
import io
import logging
from typing import Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class FrameHandler:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.frame_cache = {}
        self.cache_lock = asyncio.Lock()
        self.MAX_CACHE_SIZE = 30

    async def load_frame(self, frame_path: str, frame_version: int) -> Tuple[bytes, str]:
        """Load a frame with proper error handling and caching"""
        cache_key = f"{frame_path}_{frame_version}"
        
        async with self.cache_lock:
            if cache_key in self.frame_cache:
                return self.frame_cache[cache_key]

        try:
            loop = asyncio.get_event_loop()
            frame_data = await loop.run_in_executor(
                self.executor, self._read_and_process_frame, frame_path
            )

            async with self.cache_lock:
                if len(self.frame_cache) >= self.MAX_CACHE_SIZE:
                    oldest_keys = sorted(self.frame_cache.keys())[:len(self.frame_cache) // 2]
                    for key in oldest_keys:
                        self.frame_cache.pop(key)
                self.frame_cache[cache_key] = frame_data

            return frame_data

        except Exception as e:
            logging.error(f"Error loading frame {frame_path}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load frame: {str(e)}"
            )

    def _read_and_process_frame(self, frame_path: str) -> Tuple[bytes, str]:
        """Read and process a frame with proper error handling"""
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        try:
            with Image.open(frame_path) as img:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=95)
                content_type = 'image/jpeg'
                frame_data = img_buffer.getvalue()

        except Exception as pil_error:
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise ValueError("Failed to read frame with OpenCV")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_data = buffer.tobytes()
                content_type = 'image/jpeg'

            except Exception as cv_error:
                raise Exception(f"Failed to read frame with both PIL ({str(pil_error)}) and OpenCV ({str(cv_error)})")

        return frame_data, content_type

    async def clear_cache(self):
        """Clear the frame cache"""
        async with self.cache_lock:
            self.frame_cache.clear()

frame_handler = FrameHandler()