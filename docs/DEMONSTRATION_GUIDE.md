# Live Demonstration Guide
## AI-Based Smart Waste Detection System

This guide provides step-by-step instructions for conducting a live demonstration of the waste classification system.

---

## Pre-Demo Checklist

- [ ] Python 3.8+ installed and accessible
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Trained model available at `model/checkpoints/best_model.pth` (or via Hugging Face auto-download)
- [ ] Webcam connected and working
- [ ] Sample waste items ready (plastic bottle, paper, metal can, glass jar, food wrapper, organic item)
- [ ] Test the webcam: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

---

## Demo 1: Desktop GUI Application (Recommended)

### Launch
```bash
python desktop_app.py
```

### Demo Flow

#### Step 1 — Application Start
- The app opens with a dark-themed interface (1280×820 window).
- The model auto-downloads from Hugging Face Hub on first launch.
- Watch the status bar: "⏳ Downloading model…" → "⏳ Loading model weights…" → "Model ready on cpu".
- Console log shows timestamped progress messages.

#### Step 2 — Image Upload Classification
1. Click the **📁 Upload Image** tab.
2. Click **📂 Browse Image** → select a test image (e.g., a plastic bottle photo).
3. The image previews in the main canvas area.
4. Click **🔍 Classify** → results appear instantly in the right panel:
   - Large emoji and class name (e.g., "🔵 PLASTIC")
   - Confidence percentage (e.g., "96.2% Confidence")
   - Probability distribution bars for all 6 categories
   - Inference time in the status bar (e.g., "Inference: 45ms")

**Talking Point**: *"The model processes the image in under 100ms on a standard laptop CPU, demonstrating the efficiency of MobileNetV2's lightweight architecture."*

#### Step 3 — Live Camera Classification
1. Switch to the **📹 Live Camera** tab.
2. Click **📹 Start Camera** → live webcam feed appears.
3. Click **🔍 Start Detection** → the system begins classifying each frame.
4. Hold different waste items in front of the camera:
   - Plastic bottle → should show "PLASTIC" with high confidence
   - Paper/cardboard → should show "PAPER"
   - Metal can → should show "METAL"
5. Adjust the **Interval** slider (100ms–1000ms) to show speed vs. accuracy trade-off.
6. Adjust the **Threshold** slider to filter low-confidence predictions.
7. Click **⏸ Stop Detection** and **⏹ Stop Camera** when done.

**Talking Point**: *"Real-time detection runs at 15–30 FPS on CPU. The configurable detection interval allows balancing between responsiveness and system load."*

---

## Demo 2: CLI Webcam Application

### Launch
```bash
python app/webcam_detect.py
```

### Demo Flow
1. The app loads the model and opens the webcam.
2. Hold waste items in front of the camera.
3. The prediction and confidence score appear as an overlay on the video feed.
4. Demonstrate keyboard controls:
   - Press **S** → saves a screenshot to `screenshots/`
   - Press **SPACE** → pauses/resumes the feed
   - Press **C** → toggles confidence display
   - Press **Q** → quits the application

### Advanced Options
```bash
# Custom confidence threshold
python app/webcam_detect.py --threshold 80

# Use a different camera
python app/webcam_detect.py --camera 1

# Reduce frame skipping for faster inference
python app/webcam_detect.py --skip-frames 1
```

---

## Demo 3: Model Training (Optional Walkthrough)

Show the training pipeline end-to-end:

```bash
# 1. Download dataset
python dataset/download_data.py

# 2. Organize into categories
python dataset/organize_data.py

# 3. Split into train/val/test
python dataset/split_data.py

# 4. Train the model (15–30 min on CPU)
python model/train.py

# 5. Evaluate on test set
python model/evaluate.py
```

**Talking Point**: *"The entire pipeline — from raw data to a production-ready model — can be executed with 5 commands."*

---

## Troubleshooting During Demo

| Issue | Quick Fix |
|---|---|
| Webcam not opening | Try `--camera 1` or `--camera 2` |
| Model not found | The desktop app auto-downloads; for CLI, ensure `model/checkpoints/best_model.pth` exists |
| Laggy video feed | Increase detection interval or use `--skip-frames 5` |
| Low confidence predictions | Ensure good lighting and a plain background |
| Import errors | Run `pip install -r requirements.txt` |

---

## Key Metrics to Highlight

| Metric | Value |
|---|---|
| Test Accuracy | 98.93% |
| Inference Speed | 15–30 FPS (CPU) |
| Model Size | ~14 MB |
| Training Time | 15–30 min (CPU) |
| Categories | 6 waste types |

---

**Prepared for**: Session 8 — Live Demonstration  
**Last Updated**: February 2026
