# === Imports ===
import os
import sys
import uuid
import shutil
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

# Debug: Print environment variables
print("=== Environment Variables ===")
print(f"MODEL_CFG: {os.getenv('MODEL_CFG')}")
print(f"SAM2_PATH: {os.getenv('SAM2_PATH')}")
print(f"SAM2_CHECKPOINT: {os.getenv('SAM2_CHECKPOINT')}")
print(f"SAM2_CONFIG_PATH: {os.getenv('SAM2_CONFIG_PATH')}")
print(f"PYTHONPATH: {os.getenv('PYTHONPATH')}")
print("===========================")

import uvicorn
import base64
import cv2
import torch
import threading
import time
import logging
from PIL import Image
import colorsys
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from fastapi.responses import FileResponse
import hashlib

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Pydantic Models ===

class VideoState:
    def __init__(self, video_id, video_dir, video_filename, fps, width, height):
        self.video_id = video_id
        self.video_dir = video_dir
        self.video_filename = video_filename
        self.fps = fps
        self.width = width
        self.height = height
        self.last_access = time.time()
        self.masks_by_frame = {}         # To store mask data per frame
        self.prompts_by_obj = {}         # To store prompt data for each object
        self.predictor_states_by_obj = {}  # To store predictor states for each object
        self.obj_colors = {}             # For storing color assignments per object
        self.effects_by_obj = {}         # For storing effects per object
        self.muted_objects = set()       # To track muted object IDs
        self.fx_scripts = {}             # For loaded effect scripts
        self.tracking_complete = False    # To track object tracking status
        self.adjusted_masks_by_frame = {} # For storing adjusted mask data

class PromptData(BaseModel):
    video_id: str
    frame_idx: int = Field(..., ge=0)
    obj_id: int = Field(..., ge=1)
    points: Optional[List[List[float]]] = None
    labels: Optional[List[int]] = None
    box: Optional[List[float]] = None
    clear_old_prompts: Optional[bool] = False
    normalize_coords: Optional[bool] = False

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v: List[int]) -> List[int]:
        if any(label not in {0, 1} for label in v):
            raise ValueError('All labels must be 0 or 1')
        return v


class ExportData(BaseModel):
    video_id: str = Field(..., json_schema_extra={"examples": ["video-uuid"]})
    export_options: Optional[Dict] = Field(
        None,
        json_schema_extra={"examples": [{"video_with_effects": True}]}
    )


class DeleteObjectData(BaseModel):
    video_id: str
    obj_id: int = Field(..., ge=1, json_schema_extra={"examples": [1]})


class ResetStateData(BaseModel):
    video_id: str = Field(..., json_schema_extra={"examples": ["video-uuid"]})


class MuteObjectData(BaseModel):
    video_id: str
    obj_id: int = Field(..., ge=1, json_schema_extra={"examples": [1]})
    muted: bool = Field(..., json_schema_extra={"examples": [True]})


class EffectUploadData(BaseModel):
    video_id: str = Field(..., json_schema_extra={"examples": ["video-uuid"]})
    effect_name: str = Field(..., json_schema_extra={"examples": ["cool_effect"]})
    effect_code: str = Field(..., json_schema_extra={"examples": ["def apply(): ..."]})
    effect_config: Optional[str] = Field(
        None,
        json_schema_extra={"examples": [json.dumps({"param": "value"})]}
    )


class ApplyEffectsRequest(BaseModel):
    video_id: str
    obj_id: int
    effects: list[dict] = Field(default_factory=list)  # Allow empty list
    feather_params: dict = Field(default_factory=dict)  # Provide empty dict default
    frame_idx: int
    preview_mode: str
    reset_frame: bool
    original_frame_hash: str = Field(
        ...,
        min_length=32,
        max_length=32,
        pattern=r"^[a-f0-9]{32}$",
        json_schema_extra={"examples": ["d41d8cd98f00b204e9800998ecf8427e"]}
    )




# === Global Variables and FastAPI App Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
if not VIDEOS_DIR.exists():
    try:
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"'videos' directory created at {VIDEOS_DIR}")
    except Exception as e:
        logger.error(f"Failed to create 'videos' directory at {VIDEOS_DIR}: {e}")
        raise
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

# === Video Upload Endpoint ===
@app.post("/upload_video")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Validate file type
        if not video.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a video file."
            )

        # Generate a unique ID for the video
        video_id = str(uuid.uuid4())
        
        # Create a directory for this video
        video_dir = VIDEOS_DIR / video_id
        frames_dir = video_dir / "frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        # Save the uploaded video
        video_path = video_dir / video.filename
        try:
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save video file: {e}")
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
            raise HTTPException(status_code=500, detail="Failed to save video file")
        
        # Extract video metadata using OpenCV
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError("Failed to open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if fps <= 0 or width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions or FPS")
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
            raise HTTPException(status_code=400, detail="Invalid video file")
        
        # Extract frames
        try:
            extract_frames(str(video_path), str(frames_dir))
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
            raise HTTPException(status_code=500, detail="Failed to extract video frames")
        
        # Create video state
        video_state = VideoState(
            video_id=video_id,
            video_dir=str(video_dir),
            video_filename=video.filename,
            fps=fps,
            width=width,
            height=height
        )
        
        # Store video state
        with state_lock:
            video_states[video_id] = video_state
            
            # Save video states to file
            try:
                with open(BASE_DIR / "video_states.json", "w") as f:
                    json.dump({
                        k: {
                            "video_dir": v.video_dir,
                            "video_filename": v.video_filename,
                            "fps": v.fps,
                            "width": v.width,
                            "height": v.height,
                            "last_access": v.last_access
                        } for k, v in video_states.items()
                    }, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save video states: {e}")
                # Don't raise an exception here as the video is already processed
        
        return {
            "video_id": video_id,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": len(os.listdir(frames_dir))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during video upload: {e}")
        if 'video_dir' in locals() and os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        raise HTTPException(status_code=500, detail=str(e))

# Define preset directories for color and effect stack presets
COLOR_PRESETS_DIR = BASE_DIR / "color_presets"
EFFECT_STACK_PRESETS_DIR = BASE_DIR / "effects_stack_presets"  # ✅ Plural "effects"

# Create preset directories if they don't exist
for preset_dir in [COLOR_PRESETS_DIR, EFFECT_STACK_PRESETS_DIR]:
    if not preset_dir.exists():
        try:
            preset_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created preset directory: {preset_dir}")
        except Exception as e:
            logger.error(f"Failed to create preset directory {preset_dir}: {e}")
            raise

state_lock = threading.Lock()
video_states = {}
executor = ThreadPoolExecutor(max_workers=4)

# === SAM2 Initialization Section (Updated for SAM2.1) ===

# Add the SAM2 directory to sys.path so that Python can import SAM2 modules.
SAM2_PATH = os.getenv('SAM2_PATH')
if not SAM2_PATH:
    logger.error("SAM2_PATH environment variable not set")
    sys.exit(1)

if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    logger.error(f"Failed to import SAM2 modules: {e}")
    sys.exit(1)

# Get model config from environment
MODEL_CFG = os.getenv('MODEL_CFG', "sam2.1/sam2.1_hiera_s.yaml")

# Get checkpoint path from environment
SAM2_CHECKPOINT = os.getenv('SAM2_CHECKPOINT')
if not SAM2_CHECKPOINT:
    logger.error("SAM2_CHECKPOINT environment variable not set")
    sys.exit(1)

logger.info(f"SAM2 code path: {SAM2_PATH}")
logger.info(f"SAM2 checkpoint: {SAM2_CHECKPOINT}")
logger.info(f"Model config file (name): {MODEL_CFG}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    predictor = build_sam2_video_predictor(
        config_file=MODEL_CFG,
        ckpt_path=SAM2_CHECKPOINT,
        device=DEVICE
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=predictor,
        points_per_side=64,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )
    logger.info(f"SAM2 models initialized successfully on device: {DEVICE}.")
except Exception as e:
    logger.error(f"Failed to initialize SAM2 models: {e}")
    sys.exit(1)

# === End of SAM2 Initialization Section ===

# --- Continue with your API endpoints and helper functions ---
# (Copy the remainder of your original file from here onward without any changes.)


def compute_frame_hash(video_id, frame_idx):
    frame_path = f"videos/{video_id}/frames/{frame_idx:05d}.jpg"
    if not os.path.exists(frame_path):
        return ""
    
    with open(frame_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def run_tracking(video_id: str):
    with state_lock:
        if video_id not in video_states:
            logger.error(f"Video {video_id} not found during tracking.")
            return
        video_state = video_states[video_id]
        video_state.tracking_complete = False
        logger.info(f"Tracking started for video {video_id}")

    try:
        for obj_id, predictor_state in video_state.predictor_states_by_obj.items():
            # Run tracking for this object
            for out_frame_idx, out_obj_ids, masks in predictor.propagate_in_video(predictor_state):
                # masks is a list of masks, one per obj_id (should be only one obj_id here)
                mask = masks[0]
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask_np = (mask > 0).astype(np.uint8) * 255
                with state_lock:
                    if out_frame_idx not in video_state.masks_by_frame:
                        video_state.masks_by_frame[out_frame_idx] = {}
                    video_state.masks_by_frame[out_frame_idx][obj_id] = mask_np
                    logger.debug(f"Stored mask for object {obj_id} in frame {out_frame_idx}")

        with state_lock:
            video_state.tracking_complete = True
            logger.info(f"Tracking complete for video {video_id}")

    except Exception as e:
        logger.error(f"Error during tracking for video {video_id}: {e}")
        with state_lock:
            video_state.tracking_complete = False

def cleanup_expired_videos(expiration_seconds: int = 3600):
    while True:
        with state_lock:
            current_time = time.time()
            expired_videos = [
                video_id for video_id, state in video_states.items()
                if current_time - state.last_access > expiration_seconds
            ]
            for video_id in expired_videos:
                video_dir = video_states[video_id].video_dir
                try:
                    shutil.rmtree(video_dir)
                    del video_states[video_id]
                    logger.info(f"Cleaned up expired video {video_id}")
                except Exception as e:
                    logger.error(f"Failed to clean up video {video_id}: {e}")
        time.sleep(600)  # Run cleanup every 10 minutes

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_expired_videos, daemon=True)
cleanup_thread.start()

def non_max_suppression(masks: List[np.ndarray], iou_threshold: float = 0.5) -> List[np.ndarray]:
    if not masks:
        return []
    areas = [np.sum(mask) for mask in masks]
    sorted_indices = np.argsort(areas)[::-1]
    keep = []
    while sorted_indices.size > 0:
        current = sorted_indices[0]
        keep.append(current)
        if sorted_indices.size == 1:
            break
        rest = sorted_indices[1:]
        ious = np.array([compute_iou(masks[current], masks[i]) for i in rest])
        sorted_indices = sorted_indices[1:][ious < iou_threshold]
    return [masks[i] for i in keep]

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def load_image(frame_idx: int, video_dir: str) -> np.ndarray:
    frame_path = os.path.join(video_dir, "frames", f"{frame_idx:05d}.jpg")
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail=f"Frame {frame_idx} not found")
    image = Image.open(frame_path).convert("RGB")
    image = np.array(image)
    return image

def extract_frames(video_path: str, output_dir: str):
    try:
        import ffmpeg
    except ImportError:
        logger.error("ffmpeg-python is not installed. Install it using 'pip install ffmpeg-python'.")
        raise HTTPException(status_code=500, detail="ffmpeg-python is not installed.")
    try:
        (
            ffmpeg
            .input(video_path)
            .output(os.path.join(output_dir, '%05d.jpg'), start_number=0, qscale=2)
            .overwrite_output()
            .run()
        )
        logger.info(f"Frames extracted to {output_dir}")
    except ffmpeg.Error as e:
        stderr_output = e.stderr.decode() if e.stderr else 'No stderr available'
        logger.error(f"ffmpeg error: {stderr_output}")
        raise HTTPException(status_code=500, detail="Frame extraction failed.")

def apply_feathering(mask: np.ndarray, params: Dict) -> np.ndarray:
    import cv2
    import numpy as np

    radius = params.get('radius', 0)
    expand = params.get('expand', 0)
    opacity = params.get('opacity', 1.0)

    # Clone the mask to avoid modifying the original
    feathered_mask = mask.copy()

    # Expand or contract the mask
    if expand != 0:
        kernel_size = max(1, int(abs(expand)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if expand > 0:
            feathered_mask = cv2.dilate(feathered_mask, kernel, iterations=1)
        else:
            feathered_mask = cv2.erode(feathered_mask, kernel, iterations=1)

    # Apply Gaussian blur to feather the edges
    if radius > 0:
        feathered_mask = cv2.GaussianBlur(feathered_mask, (0, 0), radius)

    # Adjust opacity
    feathered_mask = feathered_mask * opacity

    # Ensure mask is in [0,1] range
    feathered_mask = np.clip(feathered_mask, 0, 1)

    return feathered_mask

def apply_blend_mode(base_image: np.ndarray, blend_image: np.ndarray, blend_mode: str) -> np.ndarray:
    import numpy as np

    # Ensure images are in float32 format
    base_image = base_image.astype(np.float16)
    blend_image = blend_image.astype(np.float16)

    if blend_mode == 'Normal':
        return blend_image
    elif blend_mode == 'Multiply':
        return base_image * blend_image
    elif blend_mode == 'Screen':
        return 1 - (1 - base_image) * (1 - blend_image)
    elif blend_mode == 'Overlay':
        return np.where(base_image < 0.5, 2 * base_image * blend_image, 1 - 2 * (1 - base_image) * (1 - blend_image))
    elif blend_mode == 'Add':
        return np.clip(base_image + blend_image, 0, 1)
    elif blend_mode == 'Subtract':
        return np.clip(base_image - blend_image, 0, 1)
    # Add more blend modes as needed
    else:
        logger.warning(f"Blend mode '{blend_mode}' not recognized. Using 'Normal' blend mode.")
        return blend_image

def get_param(name, default, param_type, params, min_val=None, max_val=None):
    value = params.get(name, default)
    try:
        value = param_type(value)
        if min_val is not None:
            value = max(min_val, value)
        if max_val is not None:
            value = min(max_val, value)
    except (ValueError, TypeError):
        # Log warning if logging is set up
        logger.warning(f"Invalid value for {name}. Using default: {default}")
        value = default
    return value

def blend_images(base_image, overlay_image, blend_mode):
    if blend_mode == 'Normal':
        return overlay_image
    elif blend_mode == 'Multiply':
        return base_image * overlay_image
    elif blend_mode == 'Screen':
        return 1 - (1 - base_image) * (1 - overlay_image)
    elif blend_mode == 'Overlay':
        return np.where(base_image < 0.5, 2 * base_image * overlay_image, 1 - 2 * (1 - base_image) * (1 - overlay_image))
    elif blend_mode == 'Darken':
        return np.minimum(base_image, overlay_image)
    elif blend_mode == 'Lighten':
        return np.maximum(base_image, overlay_image)
    elif blend_mode == 'Color Dodge':
        return np.where(overlay_image == 1, 1, np.minimum(base_image / (1 - overlay_image), 1))
    elif blend_mode == 'Color Burn':
        return np.where(overlay_image == 0, 0, np.maximum(1 - (1 - base_image) / overlay_image, 0))
    elif blend_mode == 'Hard Light':
        return np.where(overlay_image < 0.5, 2 * base_image * overlay_image, 1 - 2 * (1 - base_image) * (1 - overlay_image))
    elif blend_mode == 'Soft Light':
        return (1 - 2 * overlay_image) * base_image ** 2 + 2 * overlay_image * base_image
    elif blend_mode == 'Difference':
        return np.abs(base_image - overlay_image)
    elif blend_mode == 'Exclusion':
        return base_image + overlay_image - 2 * base_image * overlay_image
    # Add more blending modes as needed
    else:
        return overlay_image  # Default to normal

def apply_effects_to_frame(frame: np.ndarray, effects_data: List[Dict], video_state) -> np.ndarray:
    """
    Apply multiple objects' effects to a single frame with proper feathering at both global and effect levels
    """
    if frame is None:
        logger.error("Received null frame")
        return frame
    
    # Create a composite result frame starting with the original
    result_frame = frame.copy()
    
    # Process each object's effects
    for effect_data in effects_data:
        if not effect_data:
            continue
            
        obj_id = effect_data.get('id')
        effects = effect_data.get('effects', [])
        global_feather_params = effect_data.get('feather_params', {})
        mask = effect_data.get('mask')
        
        # Skip if missing required data
        if not all([obj_id, effects, global_feather_params, mask is not None]):
            logger.debug(f"Skipping object {obj_id} - missing required data")
            continue
        
        # Ensure mask has correct shape
        if mask.ndim == 3:
            mask = mask.squeeze()
            
        try:
            # Apply global feathering to create base mask
            global_mask = apply_feathering(mask.copy(), global_feather_params)
            if global_mask is None:
                continue
                
            if global_mask.ndim == 2:
                global_mask = global_mask[:, :, np.newaxis]
            
            # Process effects for this object
            obj_frame = result_frame.copy()
            
            for effect in effects:
                if getattr(effect, 'muted', True):
                    continue
                    
                effect_name = getattr(effect, 'name', '')
                effect_script = video_state.fx_scripts.get(effect_name)
                
                if not effect_script or 'function' not in effect_script:
                    continue
                    
                try:
                    # Get effect-specific feathering parameters
                    effect_feather_params = {
                        'radius': getattr(effect, 'radius', 0),
                        'expand': getattr(effect, 'expand', 0),
                        'opacity': getattr(effect, 'opacity', 1.0)
                    }
                    
                    # Apply effect-specific feathering to create effect mask
                    effect_mask = apply_feathering(mask.copy(), effect_feather_params)
                    if effect_mask is None:
                        continue
                        
                    if effect_mask.ndim == 2:
                        effect_mask = effect_mask[:, :, np.newaxis]
                    
                    # Combine global and effect masks
                    combined_mask = global_mask * effect_mask
                    
                    # Get effect function and parameters
                    effect_func = effect_script['function']
                    effect_params = getattr(effect, 'params', {}).copy() or {}
                    
                    # Add mask parameter if needed
                    import inspect
                    sig = inspect.signature(effect_func)
                    if 'mask' in sig.parameters:
                        effect_params['mask'] = mask
                    elif 'sam2_mask' in sig.parameters:
                        effect_params['sam2_mask'] = mask
                    
                    # Apply the effect
                    effect_result = effect_func(obj_frame.copy(), **effect_params)
                    if effect_result is None:
                        continue
                    
                    # Blend using combined mask
                    obj_frame = obj_frame * (1 - combined_mask) + effect_result * combined_mask
                    
                except Exception as e:
                    logger.error(f"Error applying effect {effect_name} to object {obj_id}: {str(e)}")
                    continue
            
            # Update result frame with this object's effects
            result_frame = obj_frame
            
        except Exception as e:
            logger.error(f"Error processing effects for object {obj_id}: {str(e)}")
            continue
    
    return result_frame




@app.post("/apply_effects")
async def apply_effects(request: ApplyEffectsRequest):
    try:
        current_hash = compute_frame_hash(request.video_id, request.frame_idx)
        if not current_hash:
            raise HTTPException(status_code=404, detail="Frame not found")
        if request.original_frame_hash != current_hash:
            raise HTTPException(
                status_code=409,
                detail="Frame content has changed since initial processing"
            )

        video_state = video_states(request.video_id)
        frame_path = os.path.join(video_state.video_dir, "frames", f"{request.frame_idx:05d}.jpg")
        original_frame_path = os.path.join(video_state.video_dir, "frames", f"original_{request.frame_idx:05d}.jpg")

        if request.reset_frame or not os.path.exists(original_frame_path):
            os.makedirs(os.path.dirname(original_frame_path), exist_ok=True)
            if os.path.exists(frame_path):
                shutil.copy2(frame_path, original_frame_path)

        frame = cv2.imread(original_frame_path if request.reset_frame else frame_path)
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to load frame data")

        effect_data = {
            "id": request.obj_id,
            "effects": request.effects,
            "feather_params": request.feather_params,
            "mask": video_state.masks_by_frame.get(request.frame_idx, {})
        }

        processed_frame = apply_effects_pipeline(
            frame,
            [effect_data],
            video_state.effect_stack
        )

        if processed_frame is not None:
            cv2.imwrite(frame_path, processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return {"status": "success", "processed": True}
        
        return {"status": "success", "processed": False}

    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Effect application failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")





def compile_final_video(video_state, temp_frames_dir: str, output_video_path: str):
    """
    Compile final video from processed frames with high quality settings
    
    Args:
        video_state: VideoState object containing video properties
        temp_frames_dir: Directory containing processed frames
        output_video_path: Path where final video will be saved
    """
    try:
        import ffmpeg
        
        # Setup input streams
        input_frames = ffmpeg.input(
            os.path.join(temp_frames_dir, '%05d.jpg'),
            framerate=video_state.fps
        )
        
        # Add audio from original video if it exists
        try:
            input_audio = ffmpeg.input(video_state.original_video_path).audio
        except Exception as e:
            logger.warning(f"Could not load audio from original video: {str(e)}")
            input_audio = None

        # Prepare output settings
        output_settings = {
            'pix_fmt': 'yuv420p',
            'vcodec': 'libx264',
            'crf': 18,            # High quality (lower = better, 18 is visually lossless)
            'preset': 'slow',     # Better compression
            's': f"{video_state.width}x{video_state.height}",
            'b:v': '5000k',       # High video bitrate
            'shortest': None      # Ensure video/audio sync
        }
        
        # Add audio settings if we have audio
        if input_audio is not None:
            output_settings['acodec'] = 'copy'  # Copy audio without re-encoding

        # Create output stream
        output_stream = ffmpeg.output(
            input_frames,
            *([] if input_audio is None else [input_audio]),
            output_video_path,
            **output_settings
        )

        # Run the compilation
        output_stream.overwrite_output().run(
            capture_stdout=True,
            capture_stderr=True
        )
        
        logger.info(
            f"Video compiled successfully at {output_video_path} "
            f"with FPS={video_state.fps}, "
            f"Resolution={video_state.width}x{video_state.height}"
        )
        
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else 'No stderr available'
        logger.error(f"ffmpeg error during compilation: {stderr}")
        raise RuntimeError(f"Video compilation failed: {stderr}")
        
    except Exception as e:
        logger.error(f"Unexpected error during video compilation: {str(e)}")
        raise

def compile_video_with_effects(video_state, output_video_path: str):
    """Compile video with all effects from all objects applied concurrently"""
    frames_dir = os.path.join(video_state.video_dir, "frames")
    temp_frames_dir = os.path.join(video_state.video_dir, "temp_frames")
    os.makedirs(temp_frames_dir, exist_ok=True)

    try:
        # Get all frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) 
                            if f.endswith('.jpg') and f.split('.')[0].isdigit()])
        
        # Collect ALL objects and their effects upfront
        logger.info("Collecting all objects and effects...")
        objects_with_effects = {}
        
        # Get all unique object IDs from masks across all frames
        all_object_ids = set()
        for frame_masks in video_state.masks_by_frame.values():
            all_object_ids.update(frame_masks.keys())
        
        logger.info(f"Found {len(all_object_ids)} objects with masks")
        
        # Collect effects and parameters for each object
        for obj_id in all_object_ids:
            effects = video_state.effects_by_obj.get(obj_id, [])
            if not effects:
                continue
                
            # Get object's feather params
            feather_params = {"radius": 0, "expand": 0, "opacity": 1.0}
            for obj in getattr(video_state, 'objects', []):
                if getattr(obj, 'id', None) == obj_id:
                    feather_params = getattr(obj, 'featherParams', feather_params)
                    break
            
            objects_with_effects[obj_id] = {
                'effects': effects,
                'feather_params': feather_params
            }
        
        logger.info(f"Found {len(objects_with_effects)} objects with effects to process")

        # Process each frame
        total_frames = len(frame_files)
        for frame_idx, frame_file in enumerate(frame_files):
            logger.info(f"Processing frame {frame_idx + 1}/{total_frames}")
            
            # Load original frame
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                logger.error(f"Could not load frame {frame_idx}")
                continue

            # Prepare effects data for this frame
            current_frame_effects = []
            frame_masks = video_state.masks_by_frame.get(frame_idx, {})
            
            for obj_id, obj_data in objects_with_effects.items():
                mask = frame_masks.get(obj_id)
                if mask is not None:
                    effect_data = {
                        'id': obj_id,
                        'effects': obj_data['effects'],
                        'feather_params': obj_data['feather_params'],
                        'mask': mask
                    }
                    current_frame_effects.append(effect_data)
            
            logger.debug(f"Frame {frame_idx + 1}: Processing {len(current_frame_effects)} objects with effects")

            try:
                # Apply all effects for this frame
                result_frame = apply_effects_to_frame(frame, current_frame_effects, video_state)
                
                # Save processed frame
                temp_frame_path = os.path.join(temp_frames_dir, frame_file)
                cv2.imwrite(temp_frame_path, result_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_idx}: {str(e)}")
                shutil.copy2(frame_path, os.path.join(temp_frames_dir, frame_file))

        logger.info("Frame processing complete. Starting video compilation...")
        compile_final_video(video_state, temp_frames_dir, output_video_path)
            
    except Exception as e:
        logger.error(f"Error during video compilation: {str(e)}")
        raise
        
    finally:
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)


@app.post("/export")
async def export_video(data: ExportData):
    video_id = data.video_id
    export_options = data.export_options or {}
    
    with state_lock:
        if video_id not in video_states:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_state = video_states[video_id]
        video_state.last_access = time.time()
        logger.info(f"Starting export for video {video_id} with options {export_options}")
    
    try:
        # Create exports directory
        output_video_dir = os.path.join(video_state.video_dir, "Exports")
        os.makedirs(output_video_dir, exist_ok=True)
        
        # Set output path
        output_video_path = os.path.join(output_video_dir, "final_video.mp4")
        
        # Export video with effects
        if export_options.get('video_with_effects', True):
            logger.info("Processing video with effects...")
            compile_video_with_effects(video_state, output_video_path)
            
        # Export masks if requested
        if export_options.get('masks', False):
            logger.info("Exporting masks...")
            export_masks_to_directory(video_state)
            
        return {
            "message": "Export completed successfully",
            "download_url": f"/videos/{video_id}/Exports/final_video.mp4"
        }
        
    except Exception as e:
        logger.error(f"Error during export: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/get_color_presets")
async def get_color_presets():
    presets = []
    try:
        for preset_file in COLOR_PRESETS_DIR.glob("*.json"):
            if preset_file.name != "last_used_preset.json":
                with preset_file.open("r", encoding="utf-8") as f:
                    preset_data = json.load(f)
                    presets.append(preset_data)
        logger.info(f"Successfully loaded {len(presets)} color presets")
        return presets
    except Exception as e:
        logger.error(f"Error loading color presets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load color presets")

@app.get("/get_last_used_color_preset")
async def get_last_used_color_preset():
    last_used_preset_path = COLOR_PRESETS_DIR / "last_used_preset.json"
    try:
        if not last_used_preset_path.exists():
            logger.info("No last used color preset found")
            return None
        with last_used_preset_path.open("r", encoding="utf-8") as f:
            preset_data = json.load(f)
            logger.info("Successfully loaded last used color preset")
            return preset_data
    except Exception as e:
        logger.error(f"Error loading last used color preset: {e}")
        return None

@app.post("/save_color_preset")
async def save_color_preset(preset_data: dict):
    preset_name = preset_data.get('preset_name')
    color_settings = preset_data.get('color_settings')

    if not preset_name or not color_settings:
        raise HTTPException(status_code=400, detail="Preset name and color settings are required")

    try:
        # Save the preset
        preset_path = COLOR_PRESETS_DIR / f"{preset_name}.json"
        with preset_path.open("w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=4)

        # Save as last used preset
        last_used_path = COLOR_PRESETS_DIR / "last_used_preset.json"
        with last_used_path.open("w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=4)

        logger.info(f"Successfully saved color preset: {preset_name}")
        return {"message": "Color preset saved successfully"}
    except Exception as e:
        logger.error(f"Error saving color preset: {e}")
        raise HTTPException(status_code=500, detail="Failed to save color preset")

@app.get("/load_color_presets")
async def load_color_presets():
    presets = []
    for preset_file in COLOR_PRESETS_DIR.glob("*.json"):
        with preset_file.open("r") as f:
            color_settings = json.load(f)
            presets.append({
                'preset_name': preset_file.stem,
                'color_settings': color_settings
            })
    return {"presets": presets}

@app.post("/save_effect_stack_preset")
async def save_effect_stack_preset(preset_data: dict):
    preset_name = preset_data.get('preset_name')
    effects_stack = preset_data.get('effects_stack')
    sub_folder = preset_data.get('sub_folder', '')

    if not preset_name or not effects_stack:
        raise HTTPException(status_code=400, detail="Preset name and effects stack are required")

    try:
        # Create subfolder if specified
        preset_dir = EFFECT_STACK_PRESETS_DIR / sub_folder
        preset_dir.mkdir(parents=True, exist_ok=True)

        # Save the preset
        preset_path = preset_dir / f"{preset_name}.json"
        with preset_path.open("w", encoding="utf-8") as f:
            json.dump(effects_stack, f, indent=4)

        logger.info(f"Successfully saved effect stack preset: {preset_name}")
        return {"message": "Effect stack preset saved successfully"}
    except Exception as e:
        logger.error(f"Error saving effect stack preset: {e}")
        raise HTTPException(status_code=500, detail="Failed to save effect stack preset")

@app.get("/load_effect_stack_presets")
async def load_effect_stack_presets():
    presets = []
    try:
        for root, dirs, files in os.walk(EFFECT_STACK_PRESETS_DIR):
            for file in files:
                if file.endswith('.json'):
                    preset_path = Path(root) / file
                    with preset_path.open("r", encoding="utf-8") as f:
                        effects_stack = json.load(f)
                        relative_path = Path(root).relative_to(EFFECT_STACK_PRESETS_DIR)
                        presets.append({
                            'preset_name': file[:-5],
                            'effects_stack': effects_stack,
                            'sub_folder': str(relative_path) if str(relative_path) != '.' else ''
                        })
        logger.info(f"Successfully loaded {len(presets)} effect stack presets")
        return {"presets": presets}
    except Exception as e:
        logger.error(f"Error loading effect stack presets: {e}")
        raise HTTPException(status_code=500, detail="Failed to load effect stack presets")

@app.post("/reset_state")
async def reset_state(data: ResetStateData):
    video_id = data.video_id
    with state_lock:
        if video_id not in video_states:
            logger.warning(f"Attempted to reset state for non-existent video {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")
        video_state = video_states[video_id]
        video_state.last_access = time.time()

        try:
            for predictor_state in video_state.predictor_states_by_obj.values():
                predictor.reset_state(predictor_state)
            logger.info(f"Reset predictor states for video {video_id}")
        except Exception as e:
            logger.error(f"Failed to reset predictor states for video {video_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to reset predictor states.")

        # Reset other state
        video_state.masks_by_frame = {}
        video_state.adjusted_masks_by_frame = {}
        video_state.prompts_by_obj = {}
        video_state.tracking_complete = False
        video_state.obj_colors = {}
        video_state.effects_by_obj = {}
        video_state.muted_objects = set()
        video_state.predictor_states_by_obj = {}  # Clear predictor states per object
        logger.info(f"Reset application state for video {video_id}")

    return {"message": "State reset"}

@app.post("/mute_object")
async def mute_object(data: MuteObjectData):
    video_id = data.video_id
    obj_id = data.obj_id
    muted = data.muted
    with state_lock:
        if video_id not in video_states:
            logger.warning(f"Attempted to mute object in non-existent video {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")
        video_state = video_states[video_id]
        video_state.last_access = time.time()
        
        if muted:
            video_state.muted_objects.add(obj_id)
            logger.info(f"Muted object {obj_id} in video {video_id}")
        else:
            video_state.muted_objects.discard(obj_id)
            logger.info(f"Unmuted object {obj_id} in video {video_id}")
    
    return {"message": f"Object {obj_id} {'muted' if muted else 'unmuted'} successfully"}

@app.get("/get_effects/{video_id}")
async def get_effects(video_id: str):
    with state_lock:
        if video_id not in video_states:
            logger.warning(f"Requested effects for non-existent video {video_id}")
            raise HTTPException(status_code=404, detail="Video not found")
        video_state = video_states[video_id]
        video_state.last_access = time.time()
    effects = []
    for effect_name, effect_data in video_state.fx_scripts.items():
        config = effect_data.get('config', {})
        effects.append({
            'name': effect_name,
            'defaultParams': config.get('defaultParams', {})
        })
    return {"effects": effects}

@app.post("/upload_effect")
async def upload_effect(data: EffectUploadData):
    effect_name = data.effect_name.strip()
    effect_code = data.effect_code
    effect_config = data.effect_config

    if not effect_name or not effect_code:
        logger.warning("Effect name and code are required.")
        raise HTTPException(status_code=400, detail="Effect name and code are required.")

    # Sanitize the effect name to prevent directory traversal
    effect_name_safe = "".join(c for c in effect_name if c.isalnum() or c in (' ', '_')).rstrip()

    # Define the path to save the effect
    fx_dir = Path(BASE_DIR) / "FX" / effect_name_safe
    fx_dir.mkdir(parents=True, exist_ok=True)

    # Save the effect code to script.py
    script_path = fx_dir / "script.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(effect_code)

    # Save the effect config to config.json if provided
    if effect_config:
        config_path = fx_dir / "config.json"
        try:
            # Validate that effect_config is valid JSON
            config_data = json.loads(effect_config)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in effect config: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON in effect config.")

    # Reload the effects in all video states (without requiring server restart)
    with state_lock:
        for video_state in video_states.values():
            video_state.load_fx_scripts()

    logger.info(f"Effect '{effect_name}' uploaded and loaded successfully.")
    return {"message": f"Effect '{effect_name}' uploaded successfully."}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
