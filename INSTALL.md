# SAM2 VFX Installation Guide

## Environment
This guide assumes you're using a Runpod container with:
- Python 3.8+
- CUDA-capable GPU
- Node.js and npm (for frontend)
- Git
- Cursor IDE

## Installation Steps

### 1. Repository Setup ✅
```bash
# Clone repositories
cd /workspace
git clone https://github.com/facebookresearch/sam2.git
git clone https://github.com/domdolo777/sam2vfx.git
```

### 2. Virtual Environment Setup ✅
```bash
# Create and activate virtual environment
cd /workspace/sam2vfx
python -m venv venv
source venv/bin/activate  # Your prompt should now show (venv)
```

### 3. SAM2 Installation (In Progress)
```bash
# Install SAM2
cd /workspace/sam2
pip install -e ".[notebooks]"  # Need to fix torch version conflict

# Download model checkpoints (Next step)
cd checkpoints
./download_ckpts.sh
```

### 4. SAM2 VFX Dependencies (Pending)
- [ ] Install core dependencies:
  - OpenCV
  - OpenFX
  - Flask/FastAPI
  - React
- [ ] Set up effect system
- [ ] Configure parallel processing

### 5. Effect System Setup (Pending)
- [ ] Create FX folder structure
- [ ] Set up effect templates
- [ ] Configure effect parameters
- [ ] Implement feathering system

### 6. Video Processing Setup (Pending)
- [ ] Configure video upload
- [ ] Set up frame extraction
- [ ] Implement mask generation
- [ ] Configure effect application

### 7. Export System Setup (Pending)
- [ ] Video export with effects
- [ ] Mask export system
- [ ] Quality preservation checks

## Current Status
- ✅ Git setup complete
- ✅ Virtual environment created
- ⏳ SAM2 installation needs torch version fix
- 📝 Documentation in progress

## Next Steps
1. Fix torch version conflict
2. Install SAM2 in virtual environment
3. Download SAM2 model checkpoints
4. Install core dependencies

## Notes
- Running in Runpod container
- GitHub authentication handled by Cursor
- Using virtual environment for clean dependency management
- Document any issues encountered
- Track performance optimizations
- Note system requirements 