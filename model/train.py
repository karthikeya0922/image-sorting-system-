"""
Training Script for Waste Classification Model
Trains MobileNetV2-based classifier on waste dataset.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model_architecture import create_model, print_model_summary
from model.config import (
    MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, 
    CHECKPOINT_CONFIG, EARLY_STOPPING_CONFIG, CLASS_NAMES, PATHS
)
from model.utils import (
    save_checkpoint, calculate_accuracy, AverageMeter, 
    EarlyStopping, plot_training_history, save_training_history
)


def get_transforms():
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(DATA_CONFIG['image_size']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((DATA_CONFIG['image_size'], DATA_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_data(train_dir, val_dir, batch_size, num_workers=2):
    """Load training and validation data."""
    train_transform, val_transform = get_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DATA_CONFIG['pin_memory']
    )
    
    return train_loader, val_loader, train_dataset.classes, train_dataset


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.1f}%'})
    
    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc_meter.avg:.1f}%'})
    
    return loss_meter.avg, acc_meter.avg


def main(args):
    """Main training function."""
    print("=" * 60)
    print("  WASTE CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Load data
    print("📁 Loading data...")
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    
    if not train_dir.exists():
        print(f"❌ Error: Training directory not found: {train_dir}")
        print("   Please run: python dataset/download_data.py")
        print("               python dataset/organize_data.py")
        print("               python dataset/split_data.py")
        return
    
    train_loader, val_loader, classes, train_dataset = load_data(
        train_dir, val_dir, 
        args.batch_size, 
        args.num_workers
    )
    
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Classes: {classes}")
    print()
    
    # Create model
    print("🧠 Creating model...")
    model = create_model(
        architecture=args.architecture,
        num_classes=len(classes),
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    print_model_summary(model)
    print()
    
    # Calculate class weights for imbalanced data
    if args.use_class_weights:
        print("⚖️  Calculating class weights for balanced training...")
        class_counts = Counter(train_dataset.targets)
        total_samples = len(train_dataset.targets)
        num_classes = len(classes)
        
        # Inverse frequency weighting
        class_weights = []
        for i in range(num_classes):
            weight = total_samples / (num_classes * class_counts[i])
            class_weights.append(weight)
            print(f"   {classes[i]}: {class_counts[i]} samples, weight={weight:.3f}")
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print()
    else:
        criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_CONFIG['patience'],
        min_delta=EARLY_STOPPING_CONFIG['min_delta'],
        mode='max'  # Maximize accuracy
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    print("🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print()
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_acc, val_loss, best_model_path)
            print(f"🏆 New best model! Val Acc: {val_acc:.2f}%")
            print()
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"⚠️  Early stopping triggered at epoch {epoch}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print("=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print()
    print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"💾 Best model saved to: {checkpoint_dir / 'best_model.pth'}")
    print()
    
    # Save training history
    results_dir = Path(PATHS['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    save_training_history(history, results_dir / 'training_history.json')
    plot_training_history(history, results_dir / 'training_history.png')
    
    print()
    print("📌 Next steps:")
    print("   1. Evaluate model: python model/evaluate.py")
    print("   2. Run webcam detection: python app/webcam_detect.py")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Waste Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-dir', type=str, default=DATA_CONFIG['train_dir'],
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, default=DATA_CONFIG['val_dir'],
                        help='Path to validation data directory')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default=MODEL_CONFIG['architecture'],
                        choices=['mobilenet_v2', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=MODEL_CONFIG['pretrained'],
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=MODEL_CONFIG['dropout'],
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=TRAIN_CONFIG['weight_decay'],
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default=TRAIN_CONFIG['optimizer'],
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--num-workers', type=int, default=DATA_CONFIG['num_workers'],
                        help='Number of data loading workers')
    
    # Device arguments
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU if available')
    parser.add_argument('--no-gpu', dest='use_gpu', action='store_false',
                        help='Force CPU training')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_CONFIG['save_dir'],
                        help='Directory to save checkpoints')
    
    # Class balancing
    parser.add_argument('--use-class-weights', action='store_true', default=False,
                        help='Use class weights to handle imbalanced data')
    parser.add_argument('--balanced', dest='use_class_weights', action='store_true',
                        help='Shorthand for --use-class-weights')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
