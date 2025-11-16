# Human Recognition Demo

Real-time webcam experience that overlays human pose skeletons, finger counts, and dominant facial emotion on top of the live feed. Built with OpenCV, MediaPipe, and DeepFace, and optimized to take advantage of GPU-accelerated DeepFace backends.

## Features

- ✅ Live webcam capture with mirrored (selfie-style) preview.
- ✅ MediaPipe Pose skeleton rendering with customizable confidence thresholds.
- ✅ MediaPipe Hands tracking with per-hand finger counting and label overlays.
- ✅ DeepFace emotion analysis (configurable detector backend, GPU-friendly).
- ✅ Graceful handling of missing detections and camera hiccups.
- ✅ Simple CLI flags to tweak camera index, resolution, and feature toggles.

## Getting Started

### 1. Prerequisites

- Python **3.11.x** (MediaPipe currently ships wheels only up to 3.11; 3.12+/3.13+ will fail to install) from [python.org](https://www.python.org/downloads/windows/). Enable *“Add Python to PATH”* during setup.
- Windows PowerShell execution policy that allows activating virtual environments. Use a process-scoped override if the default blocks `Activate.ps1`.
- A webcam and a GPU-ready PC (install NVIDIA driver + CUDA/cuDNN if you want TensorFlow GPU).
- Microsoft Visual C++ Build Tools (from the Visual Studio installer) if any dependency needs compilation.

### 2. Create and configure the environment (VS Code workflow)

1. **Create the virtual environment:**
	- Press `Ctrl+Shift+P`, run **Python: Create Environment…**, choose *Venv*, then pick the Python 3.11 interpreter.
	- VS Code generates `.venv` (or the name you select) and automatically activates it in the integrated terminal.
2. **Install dependencies:** VS Code will prompt to install from `requirements.txt`. Click *Install*, or run manually in the integrated terminal:
	```powershell
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	```
3. **Ensure VS Code uses the venv:** If it doesn’t switch automatically, run **Python: Select Interpreter** and choose the new `.venv\Scripts\python.exe` entry.

Key notes:

- The `requirements.txt` pins `tensorflow==2.19.1`, `tf-keras==2.19.0`, and `protobuf==4.25.3` to avoid Mediapipe/RetinaFace conflicts. Replacing these with newer TensorFlow wheels can break the hand/face pipelines.

- The execution-policy change is optional inside the VS Code terminal but required in a stock PowerShell where script execution is disabled by default.
- If activation still fails, skip the `Activate.ps1` step and run commands via the full path `C:\...\.venv\Scripts\python.exe`.
- After creating the venv manually, reopen VS Code (or rerun **Python: Select Interpreter**) so it uses the new interpreter.

> **GPU note:** Install the GPU-enabled wheel of TensorFlow or PyTorch before `deepface` if you want DeepFace to leverage CUDA automatically.

### 3. Verify core dependencies (optional but recommended)

```powershell
python -c "import cv2, mediapipe, deepface; print('deps-ok')"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 4. Run the demo

```powershell
python main.py --camera-index 0 --width 1280 --height 720 --emotion-interval 10
```

Press **Q** to exit the preview window. Re-run with different flags to switch webcams, resolutions, or DeepFace detector backends.

### 4. Optional flags

| Flag | Description |
| --- | --- |
| `--camera-index` | Select an alternate webcam (default `0`). |
| `--width` / `--height` | Capture resolution; reduce for faster FPS. |
| `--emotion-interval` | Analyze emotion every _N_ frames to maintain performance (default `10`). |
| `--emotion-backend` | DeepFace detector backend (`retinaface`, `mtcnn`, `opencv`, `skip`, etc.). |
| `--no-pose` / `--no-hands` | Disable pose or hand modules for debugging. |

## Implementation Notes

- The frame is flipped horizontally before processing to create a natural mirror experience.
- Finger counts adapt to MediaPipe's handedness classification and annotate near the wrist center.
- Emotion analysis uses `enforce_detection=False` to avoid crashes when no face is visible.
- All OpenCV windows and camera handles are released in a `finally` block.
- GPU acceleration is handled by DeepFace's backend (TensorFlow/PyTorch); ensure the appropriate GPU package is installed for best performance.

## Troubleshooting

- **Execution policy prevents venv activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before `.\.venv\Scripts\Activate.ps1`, or skip activation and call `.venv\Scripts\python.exe` directly.
- **MediaPipe install fails on Python 3.12/3.13+:** install Python 3.11.x instead and recreate `.venv`.
- **DeepFace warns about `tf-keras` or protobuf:** reinstall with the pinned versions (`python -m pip install --force-reinstall -r requirements.txt`).
- **Emotion label stays `Neutral`:** ensure your face is visible and well lit, keep the camera steady for a few frames, and try lowering `--emotion-interval`. The app now extracts a MediaPipe face ROI and feeds it to DeepFace; check the terminal for any `[WARN] Emotion analysis skipped ...` messages.
- **Low FPS:** Lower `--width`/`--height`, increase `--emotion-interval`, or temporarily disable pose/hands.
- **Camera busy:** Make sure no other application is using the webcam, or try another `--camera-index` value.
- **Missing CUDA support:** Install the GPU-enabled wheel of TensorFlow or PyTorch that matches your CUDA/cuDNN stack, then reinstall DeepFace.

