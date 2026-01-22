"""
Evaluation Script for Waste Classification Model
Evaluates trained model on test set and generates reports.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model_architecture import create_model
from model.config import DATA_CONFIG, MODEL_CONFIG, CLASS_NAMES, PATHS
from model.utils import (
    load_checkpoint, calculate_accuracy, 
    plot_confusion_matrix, generate_classification_report
)


def get_test_transform():
    """Get test transforms."""
    return transforms.Compose([
        transforms.Resize((DATA_CONFIG['image_size'], DATA_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
    
    accuracy = 100 * correct / total
    return accuracy, all_labels, all_preds


def main(args):
    """Main evaluation function."""
    print("=" * 60)
    print("  WASTE CLASSIFICATION MODEL EVALUATION")
    print("=" * 60)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🖥️  Device: {device}")
    print()
    
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model checkpoint not found: {model_path}")
        print("   Please train a model first: python model/train.py")
        return
    
    # Check test directory
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"❌ Error: Test directory not found: {test_dir}")
        return
    
    # Load test data
    print("📁 Loading test data...")
    test_transform = get_test_transform()
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    classes = test_dataset.classes
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Classes: {classes}")
    print()
    
    # Create and load model
    print("🧠 Loading model...")
    model = create_model(
        architecture=args.architecture,
        num_classes=len(classes),
        pretrained=False
    )
    
    load_checkpoint(model, model_path, device=device)
    model = model.to(device)
    print()
    
    # Evaluate
    print("📊 Evaluating on test set...")
    accuracy, all_labels, all_preds = evaluate(model, test_loader, device)
    
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print()
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    print()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate confusion matrix
    print("📈 Generating confusion matrix...")
    plot_confusion_matrix(
        all_labels, all_preds, classes,
        save_path=results_dir / 'confusion_matrix.png'
    )
    
    # Generate classification report
    print("📝 Generating classification report...")
    generate_classification_report(
        all_labels, all_preds, classes,
        save_path=results_dir / 'classification_report.txt'
    )
    
    print()
    print(f"✅ Results saved to: {results_dir}")
    print()
    print("📌 Next step:")
    print("   Run webcam detection: python app/webcam_detect.py")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Waste Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, 
                        default='model/checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, 
                        default=DATA_CONFIG['test_dir'],
                        help='Path to test data directory')
    parser.add_argument('--architecture', type=str, 
                        default=MODEL_CONFIG['architecture'],
                        choices=['mobilenet_v2', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, 
                        default=DATA_CONFIG['num_workers'],
                        help='Number of data loading workers')
    parser.add_argument('--results-dir', type=str,
                        default=PATHS['results_dir'],
                        help='Directory to save results')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Force CPU evaluation')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
