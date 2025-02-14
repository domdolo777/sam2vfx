# SAM2VFX Installation Guide

This guide will help you set up SAM2VFX and its dependencies correctly.

## Prerequisites

- Linux with Python ≥ 3.10
- CUDA-capable GPU
- PyTorch ≥ 2.5.1 with CUDA support
- Node.js ≥ 18.0.0 for the frontend

## Step 1: Environment Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Install SAM2

1. Clone the SAM2 repository:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
```

2. Install SAM2:
```bash
pip install -e ".[notebooks]"
```

3. Download the SAM2 checkpoint:
```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/sam2/sam2.1_hiera_small.pt
cd ..
```

## Step 3: Install SAM2VFX

1. Clone the SAM2VFX repository:
```bash
git clone https://github.com/your-org/sam2vfx.git
cd sam2vfx
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
pip install opencv-python-headless  # Headless version for server environments
pip install python-dotenv  # For environment variable management
```

3. Set up environment variables in `backend/.env`:
```plaintext
MODEL_CFG=sam2.1/sam2.1_hiera_s.yaml
SAM2_PATH=/workspace/sam2
SAM2_CHECKPOINT=/workspace/sam2/checkpoints/sam2.1_hiera_small.pt
PYTHONPATH="${SAM2_PATH}:${PYTHONPATH}"
```

4. Create symbolic link for SAM2.1 configs:
```bash
cd /workspace/sam2/sam2
ln -s configs/sam2.1 sam2.1
```

5. Install frontend dependencies:
```bash
cd ../frontend
npm install
npm install crypto-js  # Required for hash calculations
```

## Step 4: Verify Installation

1. Verify CUDA availability:
```bash
python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
```

2. Verify SAM2 installation:
```bash
python -c 'import sam2; print("SAM2 installed successfully!")'
```

3. Verify backend dependencies:
```bash
python -c 'import cv2, flask, fastapi; print("Backend dependencies installed successfully!")'
```

## Step 5: Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

## Troubleshooting

### Common Issues

1. **Missing CUDA**: Ensure CUDA is properly installed and visible to PyTorch.

2. **OpenCV Import Error**: If you encounter issues with OpenCV, try:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless
```

3. **SAM2 Config Path**: If you see "Config not found" errors:
   - Verify that the config path in `.env` points to `/workspace/sam2/sam2/configs`
   - Ensure you're using the correct config file name (`sam2.1/sam2.1_hiera_s.yaml` for SAM 2.1)
   - Check that the Hydra configuration variables are set correctly:
     ```bash
     echo $HYDRA_CONFIG_DIR
     echo $HYDRA_CONFIG_PATH
     echo $HYDRA_CONFIG_NAME
     ```
   - The config directory should be in your Python path:
     ```bash
     python -c 'import sys; print("\n".join(sys.path))'
     ```

4. **Model Loading Error**: If you see "unexpected keys" errors, ensure:
   - You're using SAM 2.1 checkpoint with SAM 2.1 config
   - The config and checkpoint versions match (both should be 2.1)

### Environment Variables

Make sure your environment variables are correctly set. You can verify them with:
```bash
echo $SAM2_PATH
echo $SAM2_CONFIG_PATH
echo $PYTHONPATH
```

### File Permissions

Ensure the application has read/write permissions for:
- The checkpoints directory
- The config directory
- The workspace directory

## Support

For additional help:
- Check the [SAM2 issues page](https://github.com/facebookresearch/sam2/issues)
- Submit an issue on the SAM2VFX repository 