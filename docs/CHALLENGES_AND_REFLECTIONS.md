# Challenges, Solutions & Reflections
## AI-Based Smart Waste Detection System

This document provides a candid reflection on the challenges encountered during the development of the waste classification system, the strategies applied to overcome them, and key lessons learned.

---

## 1. Dataset Collection & Quality

### Challenges
- **Class Imbalance**: The TrashNet dataset has uneven distribution across categories. Some categories (e.g., glass) had significantly more samples than others (e.g., organic), which can bias the model toward majority classes.
- **Corrupted Images**: A subset of downloaded images were truncated or corrupted, causing failures during training.
- **Ambiguous Categories**: Certain items (e.g., plastic-coated paper cups) are difficult to classify even for humans, leading to potential label noise.
- **Limited Dataset Size**: ~2,500 images is relatively small for training a deep learning model, increasing the risk of overfitting.

### Solutions Applied
- **Stratified Splitting**: Used stratified train/val/test split (`split_data.py`) to maintain class proportions across all sets.
- **Validation Pipeline**: `organize_data.py` includes checks for corrupted files and filters them before training.
- **Data Augmentation**: Applied extensive augmentation (flip, rotation, brightness/contrast adjustment, random crop) via `augment.py` to synthetically increase data diversity.
- **Transfer Learning**: Leveraged ImageNet pre-trained weights to compensate for limited data — the model starts with strong feature representations.

### Lessons Learned
> Transfer learning is remarkably effective for small datasets. By freezing the pre-trained backbone and only training the classifier head, we achieved 98%+ accuracy with only ~2,500 images.

---

## 2. Model Training

### Challenges
- **CPU-Only Training**: Training on CPU is 5–10× slower than GPU, making iterative experimentation time-consuming (15–30 minutes per full training run).
- **Hyperparameter Selection**: Choosing optimal learning rate, batch size, dropout rate, and scheduler configuration required multiple experiments.
- **Overfitting Risk**: With a small dataset and a powerful pre-trained model, the training accuracy can quickly reach 100% while validation accuracy plateaus — a classic overfitting scenario.
- **Learning Rate Scheduling**: Finding the right point to reduce the learning rate was critical to avoid either underfitting (too early) or oscillating (too late).

### Solutions Applied
- **Early Stopping** (`EarlyStopping` class in `model/utils.py`): Monitors validation loss with patience=7 and min_delta=0.001. Training automatically halts when no improvement is detected, preventing overfitting and saving time.
- **Dropout Regularization**: Applied dropout (p=0.2) in the classifier head to reduce overfitting.
- **StepLR Scheduler**: Reduced learning rate by a factor of 0.1 every 7 epochs, allowing the model to converge to finer minima in later epochs.
- **Best Checkpoint Saving**: Only the model with the highest validation accuracy is saved, ensuring the best generalizing weights are used for inference.
- **AverageMeter Utility**: Used to track running averages of loss and accuracy for stable training logs.

### Lessons Learned
> Early stopping is essential when training on small datasets. Without it, we observed the model reaching 100% training accuracy but declining validation performance after epoch ~15.

---

## 3. GUI Integration

### Challenges
- **UI Responsiveness**: Loading a ~14MB model blocks the UI thread for several seconds. Tkinter does not natively support async operations, making the app appear frozen during model loading.
- **Camera Thread Safety**: OpenCV's `VideoCapture` and Tkinter's main event loop are not thread-safe. Running camera capture and inference in the main thread causes severe FPS drops.
- **Image Scaling**: Displaying webcam frames and uploaded images at various resolutions while maintaining aspect ratio within the canvas was non-trivial.
- **Cross-Platform Styling**: Tkinter's appearance varies across Windows, macOS, and Linux. Achieving a consistent dark theme required extensive style configuration.

### Solutions Applied
- **Threaded Model Loading**: Model download and initialization run in a separate `threading.Thread(daemon=True)`, keeping the UI responsive with status callbacks.
- **Polling-Based Detection**: Instead of a separate thread for inference, used `root.after()` to schedule periodic inference at configurable intervals (100–1000ms), avoiding thread-safety issues.
- **Display-Inference Separation**: Video frame display runs at ~60 FPS (`root.after(16, ...)`), while inference runs at a separate, lower frequency — preserving smooth video playback.
- **Dynamic Image Scaling**: Used `PIL.Image.thumbnail()` with `LANCZOS` resampling and canvas resize event binding to maintain proper aspect ratios.
- **ttk.Style Configuration**: Set up comprehensive style definitions for all widget types using the 'clam' theme as a base, with custom colors for the dark theme palette.

### Lessons Learned
> Decoupling display refresh from inference frequency is key to building real-time ML GUI applications. Running inference on every frame is unnecessary and wasteful on CPU.

---

## 4. Deployment & Distribution

### Challenges
- **Model Distribution**: Shipping a 14MB model file with the code repository is impractical and against Git best practices.
- **Dependency Management**: PyTorch installation varies between CPU and GPU versions, and across operating systems.
- **Python Version Compatibility**: The project must work across Python 3.8–3.11 without modification.

### Solutions Applied
- **Hugging Face Hub**: The trained model is hosted on Hugging Face (`karthikeya09/smart_image_recognation`) and auto-downloaded on first launch via `hf_hub_download()`.
- **Docker Support**: A `Dockerfile` is provided for containerized deployment with all dependencies pre-installed.
- **Flexible Requirements**: `requirements.txt` uses minimum version constraints (e.g., `torch>=1.13.0`) rather than exact pinning for broader compatibility.
- **.gitignore**: Properly excludes checkpoints, datasets, cache, and virtual environments from version control.

### Lessons Learned
> Hosting model weights on Hugging Face Hub dramatically simplifies model distribution and version management compared to including them in the Git repository.

---

## 5. Real-Time Inference Optimization

### Challenges
- **CPU Latency**: Running MobileNetV2 inference on every webcam frame (30 FPS) would require <33ms per inference — achievable on GPU but tight on CPU.
- **Memory Usage**: Continuous frame capture and processing can lead to memory buildup if frames are not properly managed.

### Solutions Applied
- **Frame Skipping**: The CLI app runs inference every Nth frame (configurable via `--skip-frames`), displaying the last prediction on intermediate frames.
- **torch.no_grad()**: All inference calls are wrapped in `torch.no_grad()` to disable gradient computation, reducing memory usage and improving speed.
- **Efficient Preprocessing**: The preprocessing pipeline (`preprocess_frame()`) minimizes copies and uses in-place operations where possible.

### Lessons Learned
> For CPU-based real-time applications, frame skipping is a simple but effective strategy. Users perceive no difference between 10 FPS inference and 30 FPS inference when the display remains smooth.

---

## Summary of Key Takeaways

| Area | Key Insight |
|---|---|
| **Dataset** | Transfer learning makes small datasets viable; augmentation is essential |
| **Training** | Early stopping + dropout prevent overfitting on small data |
| **GUI** | Decouple display and inference; use threading for I/O-bound ops |
| **Deployment** | Host model weights externally; use Docker for reproducibility |
| **Optimization** | Frame skipping + `no_grad()` make CPU inference practical |

---

**Document Prepared**: February 2026  
**Session**: 8 — Reflections & Challenges
