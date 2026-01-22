"""
Webcam Detection Application
Real-time waste classification using webcam feed.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model_architecture import create_model
from model.config import DATA_CONFIG, MODEL_CONFIG, CLASS_NAMES


# Color mapping for each waste category (BGR format for OpenCV)
CATEGORY_COLORS = {
    'glass': (0, 255, 0),       # Green
    'metal': (128, 128, 128),   # Gray
    'non-recyclable': (0, 0, 0), # Black
    'organic': (0, 100, 0),     # Dark Green
    'paper': (42, 42, 165),     # Brown
    'plastic': (255, 0, 0),     # Blue
}

# Default color if category not found
DEFAULT_COLOR = (255, 255, 255)


def get_transform():
    """Get inference transforms."""
    return transforms.Compose([
        transforms.Resize((DATA_CONFIG['image_size'], DATA_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, architecture, num_classes, device):
    """Load trained model."""
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_frame(frame, transform):
    """Preprocess frame for model input."""
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply transforms
    tensor = transform(pil_image)
    
    # Add batch dimension
    return tensor.unsqueeze(0)


def predict(model, frame, transform, class_names, device):
    """Make prediction on frame."""
    # Preprocess
    input_tensor = preprocess_frame(frame, transform).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = class_names[predicted.item()]
    confidence = confidence.item() * 100
    
    return class_name, confidence


def draw_prediction(frame, class_name, confidence, fps, paused=False, threshold=50.0):
    """Draw prediction overlay on frame."""
    height, width = frame.shape[:2]
    
    # Get category color
    color = CATEGORY_COLORS.get(class_name.lower(), DEFAULT_COLOR)
    
    # Draw colored border
    border_size = 10
    cv2.rectangle(frame, (0, 0), (width, height), color, border_size)
    
    # Create semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw class name
    if confidence >= threshold:
        text = class_name.upper()
        text_color = color
    else:
        text = "DETECTING..."
        text_color = (128, 128, 128)
    
    cv2.putText(frame, text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    
    # Draw confidence
    conf_text = f"Confidence: {confidence:.1f}%"
    cv2.putText(frame, conf_text, (20, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw FPS in bottom left
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw controls hint in bottom right
    controls = "Q: Quit | S: Screenshot | SPACE: Pause"
    text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(frame, controls, (width - text_size[0] - 20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw PAUSED indicator if paused
    if paused:
        cv2.putText(frame, "PAUSED", (width // 2 - 80, height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    return frame


def save_screenshot(frame, output_dir):
    """Save screenshot with timestamp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"screenshot_{timestamp}.jpg"
    
    cv2.imwrite(str(filename), frame)
    print(f"📸 Screenshot saved: {filename}")
    
    return filename


def main(args):
    """Main webcam detection function."""
    print("=" * 60)
    print("  WASTE CLASSIFICATION - WEBCAM DETECTION")
    print("=" * 60)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model checkpoint not found: {model_path}")
        print("   Please train a model first: python model/train.py")
        return
    
    # Load model
    print("🧠 Loading model...")
    class_names = CLASS_NAMES
    model = load_model(model_path, args.architecture, len(class_names), device)
    print(f"   Classes: {class_names}")
    print()
    
    # Get transforms
    transform = get_transform()
    
    # Open webcam
    print(f"📹 Opening webcam (camera {args.camera})...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open webcam (camera {args.camera})")
        print("   Try a different camera index: --camera 1")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print()
    print("✅ Webcam opened successfully!")
    print()
    print("=" * 40)
    print("  CONTROLS")
    print("=" * 40)
    print("  Q      - Quit")
    print("  S      - Save screenshot")
    print("  SPACE  - Pause/Resume")
    print("  C      - Toggle confidence display")
    print("=" * 40)
    print()
    print("🚀 Starting detection... Press 'Q' to quit.")
    print()
    
    # State variables
    paused = False
    show_confidence = True
    frame_count = 0
    fps = 0
    prev_time = time.time()
    
    # Inference frequency (skip frames for better performance on CPU)
    inference_every = args.skip_frames
    last_prediction = ("DETECTING...", 0.0)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to read frame from webcam")
                break
            
            if not paused:
                frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - prev_time >= 1.0:
                    fps = frame_count / (current_time - prev_time)
                    frame_count = 0
                    prev_time = current_time
                
                # Run inference (not every frame for better performance)
                if frame_count % inference_every == 0:
                    class_name, confidence = predict(model, frame, transform, class_names, device)
                    last_prediction = (class_name, confidence)
            
            # Draw prediction on frame
            display_frame = draw_prediction(
                frame.copy(), 
                last_prediction[0], 
                last_prediction[1], 
                fps, 
                paused,
                args.threshold
            )
            
            # Show frame
            cv2.imshow('Waste Classification', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("👋 Quitting...")
                break
            elif key == ord('s') or key == ord('S'):
                save_screenshot(display_frame, args.save_dir)
            elif key == ord(' '):
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord('c') or key == ord('C'):
                show_confidence = not show_confidence
                print(f"📊 Confidence display: {'ON' if show_confidence else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam released. Goodbye!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Waste Classification Webcam Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, 
                        default='model/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--architecture', type=str, 
                        default=MODEL_CONFIG['architecture'],
                        choices=['mobilenet_v2', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index')
    parser.add_argument('--threshold', type=float, default=50.0,
                        help='Confidence threshold for displaying prediction')
    parser.add_argument('--skip-frames', type=int, default=3,
                        help='Run inference every N frames (higher = faster but less responsive)')
    parser.add_argument('--save-dir', type=str, default='screenshots',
                        help='Directory to save screenshots')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Force CPU inference')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
