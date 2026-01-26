"""
Upload Fine-tuned Waste Classification Model to Hugging Face Hub
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# Configuration - Update these values
REPO_NAME = "waste-classification-mobilenetv2"  # Change to your desired repo name
MODEL_PATH = "model/checkpoints/best_model.pth"
RESULTS_DIR = "model/results"

# Model information
MODEL_INFO = {
    "model_name": "Waste Classification MobileNetV2",
    "architecture": "MobileNetV2",
    "num_classes": 6,
    "classes": ["glass", "metal", "non-recyclable", "organic", "paper", "plastic"],
    "image_size": 224,
    "framework": "PyTorch",
    "accuracy": "97.46%",
    "description": "Fine-tuned MobileNetV2 model for waste classification into 6 categories"
}


def create_model_card():
    """Create a README.md model card for Hugging Face."""
    model_card = f"""---
tags:
- image-classification
- pytorch
- waste-classification
- mobilenetv2
- computer-vision
license: mit
datasets:
- custom
metrics:
- accuracy
---

# 🗑️ Waste Classification Model (MobileNetV2)

A fine-tuned MobileNetV2 model for classifying waste items into 6 categories using computer vision.

## Model Description

This model classifies images of waste items into the following categories:

| Category | Description |
|----------|-------------|
| 🔵 **Plastic** | Bottles, bags, containers |
| 📄 **Paper** | Newspapers, cardboard, magazines |
| 🔘 **Metal** | Cans, foil, metal containers |
| 💚 **Glass** | Bottles, jars |
| 🟢 **Organic** | Food waste, plant matter |
| ⚫ **Non-recyclable** | Mixed waste, contaminated items |

## Model Details

- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Framework**: PyTorch
- **Input Size**: 224x224 RGB images
- **Output**: 6 class probabilities
- **Validation Accuracy**: {MODEL_INFO['accuracy']}

## Usage

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 6)
checkpoint = torch.load('best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
image = Image.open('your_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

classes = ['glass', 'metal', 'non-recyclable', 'organic', 'paper', 'plastic']
print(f'Predicted: {{classes[predicted.item()]}}')
```

## Training Details

- **Dataset**: ~21,000 waste images
- **Training Split**: 70% train, 15% validation, 15% test
- **Optimizer**: Adam with learning rate 0.001
- **Class Weights**: Used to handle class imbalance
- **Data Augmentation**: Random crop, horizontal flip, rotation, color jitter

## License

MIT License
"""
    return model_card


def upload_to_huggingface(username: str):
    """Upload model to Hugging Face Hub."""
    
    print("=" * 60)
    print("  UPLOAD MODEL TO HUGGING FACE")
    print("=" * 60)
    print()
    
    # Full repository name
    repo_id = f"{username}/{REPO_NAME}"
    print(f"📦 Repository: {repo_id}")
    print()
    
    # Initialize API
    api = HfApi()
    
    # Create repository
    print("📁 Creating repository...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"   ✅ Repository created/exists: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"   ⚠️  Repository issue: {e}")
    print()
    
    # Create and upload model card
    print("📝 Creating model card...")
    model_card = create_model_card()
    readme_path = Path("README_HF.md")
    readme_path.write_text(model_card)
    
    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    readme_path.unlink()  # Clean up
    print("   ✅ Model card uploaded")
    print()
    
    # Upload model checkpoint
    print("🧠 Uploading model checkpoint...")
    upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo="best_model.pth",
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"   ✅ Model uploaded: best_model.pth")
    print()
    
    # Upload model info
    print("📋 Uploading model info...")
    config_path = Path("config.json")
    config_path.write_text(json.dumps(MODEL_INFO, indent=2))
    
    upload_file(
        path_or_fileobj=str(config_path),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model"
    )
    config_path.unlink()  # Clean up
    print("   ✅ Config uploaded: config.json")
    print()
    
    # Upload training history if exists
    history_path = Path(RESULTS_DIR) / "training_history.json"
    if history_path.exists():
        print("📊 Uploading training history...")
        upload_file(
            path_or_fileobj=str(history_path),
            path_in_repo="training_history.json",
            repo_id=repo_id,
            repo_type="model"
        )
        print("   ✅ Training history uploaded")
        print()
    
    # Upload training plot if exists
    plot_path = Path(RESULTS_DIR) / "training_history.png"
    if plot_path.exists():
        print("📈 Uploading training plot...")
        upload_file(
            path_or_fileobj=str(plot_path),
            path_in_repo="training_history.png",
            repo_id=repo_id,
            repo_type="model"
        )
        print("   ✅ Training plot uploaded")
        print()
    
    print("=" * 60)
    print("  UPLOAD COMPLETE!")
    print("=" * 60)
    print()
    print(f"🎉 Model available at: https://huggingface.co/{repo_id}")
    print()
    print("📌 To use your model:")
    print(f"   from huggingface_hub import hf_hub_download")
    print(f"   model_path = hf_hub_download(repo_id='{repo_id}', filename='best_model.pth')")
    print()


if __name__ == "__main__":
    print()
    print("🔐 First, login to Hugging Face:")
    print("   Run: huggingface-cli login")
    print()
    
    username = input("Enter your Hugging Face username: ").strip()
    
    if username:
        upload_to_huggingface(username)
    else:
        print("❌ Username required!")
