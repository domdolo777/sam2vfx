Below is a final, comprehensive installation and configuration guide that covers everything—from cloning repositories to setting up your virtual environment and ensuring the SAM 2 model files are correctly referenced. This guide assumes that your combined repository structure is as follows:

```
/workspace/sam2/sam2vfx/
├── backend/           # Your Flask backend code (e.g., main.py, frame_handler.py, etc.)
├── frontend/          # Your React frontend code (contains package.json, public/, src/, etc.)
└── sam2/              # SAM 2 repository (cloned from facebookresearch/sam2)
    ├── checkpoints/   # Model checkpoints (e.g., sam2.1_hiera_small.pt, etc.)
    ├── configs/       # Configuration files for SAM 2
    └── ...            # Other SAM 2 source files
```

In your case, the SAM 2 checkpoints have been downloaded to  
`/workspace/sam2/sam2vfx/sam2/checkpoints/`  
and you observed that the small model checkpoint is named **`sam2.1_hiera_small.pt`** (not `sam2_hiera_small.pt`).

Follow the steps below to ensure everything is set up correctly.

---

## 1. Repository Setup

### A. Clone the SAM2VFX Repository

From your working directory, run:
```bash
cd /workspace/sam2
git clone https://github.com/domdolo777/sam2vfx.git
```
This creates the directory:  
**`/workspace/sam2/sam2vfx`**

### B. Integrate the SAM 2 Repository

Place the SAM 2 code inside your SAM2VFX project. For example, if you want SAM 2 as a subfolder inside SAM2VFX, you can do one of the following:

#### Option 1: Git Submodule
```bash
cd /workspace/sam2/sam2vfx
git submodule add https://github.com/facebookresearch/sam2.git sam2
git submodule update --init --recursive
```

#### Option 2: Physically Move the SAM 2 Directory
If you already have a SAM2 clone at `/workspace/sam2/sam2`, move it:
```bash
mv /workspace/sam2/sam2 /workspace/sam2/sam2vfx/sam2
```

After this step, your SAM 2 code should be located at:  
**`/workspace/sam2/sam2vfx/sam2`**

---

## 2. Python Virtual Environment (for Backend & SAM 2)

### A. Create & Activate a Virtual Environment

It’s best to create a single virtual environment for all your Python code. For example, create it in your backend folder:
```bash
cd /workspace/sam2/sam2vfx/backend
python3 -m venv env
source env/bin/activate
```
Your prompt should now show `(env)`.

### B. Create a `requirements.txt` for the Backend

In `/workspace/sam2/sam2vfx/backend/requirements.txt`, include at minimum:
```txt
Flask
flask-cors
opencv-python
numpy
torch>=2.5.1
Pillow
tqdm
hydra-core
fastapi
uvicorn
```
Add any other packages your backend code requires.

### C. Install the Backend Dependencies

From within the activated virtual environment:
```bash
pip install -r requirements.txt
```

---

## 3. Install SAM 2 Dependencies

### A. Navigate to the SAM 2 Folder

```bash
cd /workspace/sam2/sam2vfx/sam2
```

### B. Install SAM 2 in Editable Mode (with Notebook Extras)

```bash
pip install -e ".[notebooks]"
```

### C. Download the SAM 2 Model Checkpoints

Navigate into the checkpoints directory and run the download script:
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```
After this, verify that the file **`sam2.1_hiera_small.pt`** exists in:
```bash
ls /workspace/sam2/sam2vfx/sam2/checkpoints
```

---

## 4. Set Environment Variables

From your SAM2VFX project root (which is `/workspace/sam2/sam2vfx`), set the following environment variables so that your backend code can find the SAM 2 modules, checkpoints, and config files:

```bash
cd /workspace/sam2/sam2vfx
export SAM2_PATH=$(pwd)/sam2
export SAM2_CHECKPOINT=$(python -c "import os; print(os.path.join(os.environ.get('SAM2_PATH'), 'checkpoints', 'sam2.1_hiera_small.pt'))")
export SAM2_CONFIG_PATH=$(python -c "import os; print(os.path.join(os.environ.get('SAM2_PATH'), 'configs'))")
export PYTHONPATH="${SAM2_PATH}:${PYTHONPATH}"
```

*Tip:* Add these lines to your shell profile (e.g., `~/.bashrc`) if you want them automatically set.

---

## 5. Update Your Backend Code

In your backend’s main file (located at `/workspace/sam2/sam2vfx/backend/main.py`), update the SAM 2 path settings. For example, change any lines like:

```python
SAM2_CHECKPOINT = "/mnt/c/Users/cryst/segment-anything-2/checkpoints/sam2_hiera_small.pt"
SAM2_CONFIG_PATH = "/mnt/c/Users/cryst/segment-anything-2/sam2_configs"
SAM2_PATH = os.getenv('SAM2_PATH', '/mnt/c/Users/cryst/segment-anything-2')
```

to:

```python
import os

SAM2_PATH = os.getenv('SAM2_PATH', '/workspace/sam2/sam2vfx/sam2')
SAM2_CHECKPOINT = os.path.join(SAM2_PATH, "checkpoints", "sam2.1_hiera_small.pt")
SAM2_CONFIG_PATH = os.path.join(SAM2_PATH, "configs")
```

This ensures your backend loads the model from `/workspace/sam2/sam2vfx/sam2/checkpoints/sam2.1_hiera_small.pt`.

---

## 6. Run the Backend Server

With your virtual environment still active, navigate to your backend folder:
```bash
cd /workspace/sam2/sam2vfx/backend
```
Then run your backend server (e.g., using Flask or FastAPI):
```bash
python main.py
```
Your logs should now indicate that SAM 2 is loading the model from the correct checkpoint path.

---

## 7. Frontend Setup

### A. Install Node.js and npm

If npm isn’t installed in your container, install it. For Ubuntu, you can run:
```bash
apt update && apt install -y nodejs npm
```
Verify the installation:
```bash
node -v
npm -v
```

### B. Navigate to the Frontend Folder

From your project root:
```bash
cd /workspace/sam2/sam2vfx/frontend
```

### C. Install Frontend Dependencies

```bash
npm install
```

### D. Start the Frontend Development Server

```bash
npm start
```
Your React app should now open in your browser. Ensure the API endpoints in your frontend code point to your backend (for example, `http://<your-ip>:5000`).

---

## 8. Final Verification

1. **Test SAM2 Module Import:**  
   In your activated virtual environment (from anywhere inside `/workspace/sam2/sam2vfx`), run:
   ```bash
   python -c "import sam2; print(sam2.__file__)"
   ```
   You should see a path like `/workspace/sam2/sam2vfx/sam2/sam2/__init__.py`.

2. **Check Environment Variables:**
   ```bash
   echo $SAM2_PATH
   echo $SAM2_CHECKPOINT
   echo $SAM2_CONFIG_PATH
   echo $PYTHONPATH
   ```
   These should reflect the correct paths (e.g., `/workspace/sam2/sam2vfx/sam2/...`).

3. **Test the Backend API:**  
   Use a tool like curl or Postman to send a request to your API endpoint (e.g., `/predict` if implemented) and verify that it returns the expected response.

4. **Test the Frontend:**  
   Confirm that your React app loads and correctly interacts with your backend.

---

## Summary

1. **Repository Setup:**
   - Clone SAM2VFX into `/workspace/sam2/sam2vfx`.
   - Clone or move the SAM2 repository into `/workspace/sam2/sam2vfx/sam2`.

2. **Python Environment:**
   - Create and activate a virtual environment in `/workspace/sam2/sam2vfx/backend/env`.
   - Install backend dependencies via `pip install -r requirements.txt`.

3. **SAM2 Installation:**
   - In `/workspace/sam2/sam2vfx/sam2`, run `pip install -e ".[notebooks]"`.
   - Download checkpoints with `cd checkpoints && ./download_ckpts.sh`.

4. **Set Environment Variables (from `/workspace/sam2/sam2vfx`):**
   ```bash
   export SAM2_PATH=$(pwd)/sam2
   export SAM2_CHECKPOINT=$(python -c "import os; print(os.path.join(os.environ.get('SAM2_PATH'), 'checkpoints', 'sam2.1_hiera_small.pt'))")
   export SAM2_CONFIG_PATH=$(python -c "import os; print(os.path.join(os.environ.get('SAM2_PATH'), 'configs'))")
   export PYTHONPATH="${SAM2_PATH}:${PYTHONPATH}"
   ```

5. **Update Backend Code:**  
   In `backend/main.py`, use:
   ```python
   import os
   SAM2_PATH = os.getenv('SAM2_PATH', '/workspace/sam2/sam2vfx/sam2')
   SAM2_CHECKPOINT = os.path.join(SAM2_PATH, "checkpoints", "sam2.1_hiera_small.pt")
   SAM2_CONFIG_PATH = os.path.join(SAM2_PATH, "configs")
   ```

6. **Run the Backend:**  
   In `/workspace/sam2/sam2vfx/backend`, run:
   ```bash
   python main.py
   ```

7. **Frontend Setup:**  
   - Install Node.js/npm if needed.
   - In `/workspace/sam2/sam2vfx/frontend`, run:
     ```bash
     npm install
     npm start
     ```

Following these steps will ensure that all dependencies are correctly installed, the SAM2 code is in the right location, and your backend references the correct checkpoint and config paths.

If you have any further questions or need additional assistance, please let me know!
