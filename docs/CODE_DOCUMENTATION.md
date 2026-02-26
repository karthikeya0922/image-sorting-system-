# Code Documentation
## AI-Based Smart Waste Detection System

Complete reference for every module, class, and function in the codebase.

---

## Table of Contents

1. [Dataset Scripts](#1-dataset-scripts)
2. [Model Modules](#2-model-modules)
3. [Webcam Application](#3-webcam-application)
4. [Desktop GUI Application](#4-desktop-gui-application)
5. [Test Suite](#5-test-suite)
6. [Configuration Files](#6-configuration-files)

---

## 1. Dataset Scripts

### `dataset/download_data.py`
Downloads the TrashNet dataset from GitHub.

| Function | Description |
|---|---|
| Automated download | Fetches ~2,500 images from the TrashNet repository |
| Progress tracking | Shows download progress with tqdm progress bar |
| Integrity validation | Verifies downloaded files are not corrupted |
| Output | Saves to `dataset/raw/` |

---

### `dataset/organize_data.py`
Organizes raw images into category folders.

| Function | Description |
|---|---|
| Category mapping | Maps TrashNet categories to 6 target classes |
| Image counting | Reports per-category image counts |
| Corruption check | Filters out unreadable/corrupted images |
| Output | Creates `dataset/processed/` with category subfolders |

---

### `dataset/augment.py`
Implements data augmentation using the Albumentations library.

| Transform | Probability | Details |
|---|---|---|
| Horizontal Flip | 0.5 | Mirrors image horizontally |
| Rotation | 0.5 | Rotates ±15° |
| ColorJitter | 0.3 | Adjusts brightness, contrast, saturation |
| Random Crop & Resize | — | Crops and resizes to 224×224 |

---

### `dataset/split_data.py`
Splits the organized dataset into train/validation/test sets.

| Property | Value |
|---|---|
| Split ratio | 70% train / 15% val / 15% test |
| Stratification | Yes (maintains class proportions) |
| Random seed | 42 (reproducibility) |
| Output | `dataset/processed/train/`, `val/`, `test/` |

---

## 2. Model Modules

### `model/model_architecture.py`

#### Class: `WasteClassifier(nn.Module)`
MobileNetV2-based waste classifier.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(num_classes=6, pretrained=True, dropout=0.2)` | Loads MobileNetV2, replaces classifier head with Dropout + Linear |
| `forward` | `(x) → Tensor` | Forward pass. Input: (B, 3, 224, 224). Output: (B, num_classes) |

#### Class: `WasteClassifierResNet(nn.Module)`
Alternative ResNet18-based classifier.

| Method | Signature | Description |
|---|---|---|
| `__init__` | `(num_classes=6, pretrained=True, dropout=0.2)` | Loads ResNet18, replaces final FC layer |
| `forward` | `(x) → Tensor` | Forward pass |

#### Functions

| Function | Signature | Description |
|---|---|---|
| `create_model` | `(architecture, num_classes, pretrained, dropout) → nn.Module` | Factory function: creates MobileNetV2 or ResNet18 model |
| `count_parameters` | `(model) → int` | Counts trainable parameters |
| `print_model_summary` | `(model) → None` | Prints architecture, parameter count |

---

### `model/config.py`
Contains all hyperparameters and configuration dictionaries.

| Config Dict | Key Parameters |
|---|---|
| `MODEL_CONFIG` | architecture, num_classes (6), pretrained, dropout (0.2) |
| `TRAIN_CONFIG` | batch_size (32), num_epochs (20), learning_rate (0.001), optimizer (adam) |
| `SCHEDULER_CONFIG` | type (step), step_size (7), gamma (0.1), patience (3) |
| `DATA_CONFIG` | image_size (224), num_workers (2), train/val/test directory paths |
| `CHECKPOINT_CONFIG` | save_dir, save_every (5), save_best_only (True) |
| `EARLY_STOPPING_CONFIG` | enabled (True), patience (7), min_delta (0.001) |
| `CLASS_NAMES` | ['glass', 'metal', 'non-recyclable', 'organic', 'paper', 'plastic'] |
| `PATHS` | Aggregated path references for train, val, test, checkpoints, results |

---

### `model/train.py`
Main training script with CLI argument support.

| Function | Signature | Description |
|---|---|---|
| `get_transforms` | `() → (train_transform, val_transform)` | Returns augmented train and standard val transforms |
| `load_data` | `(train_dir, val_dir, batch_size, num_workers)` | Creates train/val DataLoaders with ImageFolder |
| `train_one_epoch` | `(model, train_loader, criterion, optimizer, device)` | Single epoch: forward, loss, backward, step |
| `validate` | `(model, val_loader, criterion, device)` | Evaluates on validation set |
| `main` | `(args)` | Full training pipeline: load data → train → save best → plot history |
| `parse_args` | `()` | CLI argument parser (epochs, lr, batch-size, etc.) |

---

### `model/evaluate.py`
Evaluates the trained model and generates reports.

| Function | Signature | Description |
|---|---|---|
| `get_test_transform` | `()` | Returns standard test transform (resize, normalize) |
| `evaluate` | `(model, test_loader, device)` | Returns accuracy, all labels, all predictions |
| `main` | `(args)` | Loads model → evaluates → generates confusion matrix + classification report |
| `parse_args` | `()` | CLI argument parser (model path, test-dir, architecture) |

---

### `model/utils.py`
Helper functions for training, evaluation, and visualization.

| Function/Class | Signature | Description |
|---|---|---|
| `save_checkpoint` | `(model, optimizer, epoch, val_acc, val_loss, filepath)` | Saves model + optimizer state dict |
| `load_checkpoint` | `(model, filepath, optimizer=None, device)` | Loads model state, returns epoch number |
| `calculate_accuracy` | `(outputs, labels) → float` | Computes accuracy (0–100) from logits |
| `plot_training_history` | `(history, save_path)` | Generates loss + accuracy curve plots |
| `plot_confusion_matrix` | `(y_true, y_pred, class_names, save_path)` | Generates seaborn heatmap confusion matrix |
| `generate_classification_report` | `(y_true, y_pred, class_names, save_path)` | Generates sklearn classification report |
| `save_training_history` | `(history, save_path)` | Saves history dict to JSON |
| `AverageMeter` | Class | Computes running average of a metric |
| `EarlyStopping` | Class (`patience, min_delta, mode`) | Stops training when validation metric stops improving |

---

## 3. Webcam Application

### `app/webcam_detect.py`
CLI-based real-time webcam detection.

| Function | Signature | Description |
|---|---|---|
| `get_transform` | `()` | Returns inference transforms (resize, normalize) |
| `load_model` | `(model_path, architecture, num_classes, device)` | Loads trained model from checkpoint |
| `preprocess_frame` | `(frame, transform) → Tensor` | Converts BGR frame to normalized tensor |
| `predict` | `(model, frame, transform, class_names, device)` | Returns (class_name, confidence) |
| `draw_prediction` | `(frame, class_name, confidence, fps, paused, threshold)` | Draws overlay with prediction, confidence, FPS, controls hint |
| `save_screenshot` | `(frame, output_dir) → Path` | Saves timestamped screenshot |
| `main` | `(args)` | Main loop: capture → predict → draw → display → handle keys |

#### Constants
- `CATEGORY_COLORS`: BGR color mapping for each waste category
- Keyboard controls: Q (quit), S (screenshot), SPACE (pause), C (toggle confidence)

---

## 4. Desktop GUI Application

### `desktop_app.py`

#### Class: `WasteModel`
Wraps model downloading, loading, and inference.

| Method | Description |
|---|---|
| `__init__()` | Initializes device, transform pipeline |
| `load(status_callback)` | Downloads model from Hugging Face Hub, loads weights |
| `is_loaded` | Property: returns True if model is ready |
| `predict_pil(pil_img)` | Classifies a PIL image → returns dict with class, confidence, probabilities |
| `predict_cv(cv_frame)` | Classifies an OpenCV BGR frame (converts to PIL internally) |

#### Class: `WasteClassifierApp`
Main Tkinter GUI application.

| Method Group | Methods | Description |
|---|---|---|
| **Setup** | `__init__`, `_setup_styles`, `_build_ui` | Window initialization, ttk style configuration, layout construction |
| **Upload Tab** | `_build_upload_tab`, `_browse_image`, `_display_uploaded_image`, `_classify_uploaded` | Image file browser, canvas preview, single-image classification |
| **Camera Tab** | `_build_camera_tab`, `_start_camera`, `_stop_camera`, `_update_video_frame` | Camera lifecycle management, live frame display at ~60 FPS |
| **Detection** | `_toggle_detection`, `_start_detection`, `_stop_detection`, `_run_detection` | Periodic inference using `root.after()` at configurable interval |
| **Results** | `_build_results_panel`, `_update_results` | Updates emoji, class label, confidence, probability bars |
| **Settings** | `_on_interval_change`, `_on_threshold_change` | Event handlers for interval/threshold controls |
| **Utilities** | `_log`, `_on_close`, `_build_status_bar`, `_load_model_thread` | Console logging, cleanup, status indicators, threaded model loading |

#### Constants
- `REPO_ID`: Hugging Face model repository (`karthikeya09/smart_image_recognation`)
- `CLASSES`: 6 waste categories
- `CLASS_INFO`: Emoji, color, and description per class
- Dark theme palette: `BG_DARK`, `BG_CARD`, `BG_INPUT`, `FG_TEXT`, `ACCENT`, etc.

---

## 5. Test Suite

### `test_project.py`
Validates project structure and code integrity.

| Function | Description |
|---|---|
| `check_file_exists(filepath)` | Verifies a file exists on disk |
| `check_python_syntax(filepath)` | Compiles Python file to check for syntax errors |
| `test_project_structure()` | Checks all 13+ required files are present |
| `test_python_syntax()` | Validates syntax of all 7 Python source files |
| `test_model_architecture()` | Checks for WasteClassifier class, forward method, create_model function |
| `test_config()` | Verifies MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, CLASS_NAMES exist |
| `test_readme()` | Checks README.md for essential sections (Overview, Installation, etc.) |
| `main()` | Runs all tests and prints summary |

---

## 6. Configuration Files

### `requirements.txt`
Python dependencies with minimum version constraints.

### `Dockerfile`
Docker container configuration for deployment.

### `.gitignore`
Excludes: datasets, model checkpoints, `__pycache__`, virtual environments, IDE files.

### `vercel.json`
Configuration for Vercel deployment (web app).

---

**Document Prepared**: February 2026  
**Total Source Files**: 15+  
**Total Lines of Code**: ~2,500+
