"""
Upload README and config to Hugging Face
"""

from huggingface_hub import HfApi
import json

import os

# Use environment variable or prompt for token
TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    from huggingface_hub import login
    login()
    TOKEN = True # login() sets the token globally for the session

REPO_ID = "karthikeya09/smart_image_recognation"

api = HfApi(token=None if TOKEN is True else TOKEN)

# Create README
readme = """---
tags:
- image-classification
- pytorch
- waste-classification
- mobilenetv2
- computer-vision
- recycling
license: mit
metrics:
- accuracy
pipeline_tag: image-classification
---

# 🗑️ Smart Waste Classification Model

A fine-tuned **MobileNetV2** model for classifying waste items into 6 categories using computer vision.

## Model Performance
- **Validation Accuracy**: 97.46%
- **Framework**: PyTorch
- **Architecture**: MobileNetV2

## Classes

| Class | Description | Color |
|-------|-------------|-------|
| 🔵 **plastic** | Bottles, bags, containers | Blue |
| 📄 **paper** | Newspapers, cardboard, magazines | Brown |
| 🔘 **metal** | Cans, foil, batteries | Gray |
| 💚 **glass** | Bottles, jars | Green |
| 🟢 **organic** | Food waste, plant matter | Dark Green |
| ⚫ **non-recyclable** | Mixed/contaminated waste | Black |

## Quick Usage

```python
import torch
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(repo_id="karthikeya09/smart_image_recognation", filename="best_model.pth")

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1280, 6)
)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict
classes = ['glass', 'metal', 'non-recyclable', 'organic', 'paper', 'plastic']
image = Image.open('your_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

print(f'Predicted: {classes[predicted.item()]} ({confidence.item()*100:.1f}%)')
```

## Training Details

- **Dataset**: ~21,000 waste images
- **Training Split**: 70% train, 15% val, 15% test
- **Optimizer**: Adam (lr=0.001)
- **Class Weights**: Used to handle class imbalance
- **Data Augmentation**: Random crop, flip, rotation, color jitter
- **Input Size**: 224x224 RGB

## Dataset Distribution

| Category | Images |
|----------|--------|
| Organic | 6,620 |
| Glass | 4,022 |
| Paper | 3,882 |
| Metal | 3,428 |
| Plastic | 1,870 |
| Non-recyclable | 1,394 |

## Model Architecture

```
MobileNetV2 (pretrained on ImageNet)
└── classifier
    ├── Dropout(p=0.2)
    └── Linear(1280, 6)
```

## License

MIT License

## Author

**K Karthikeya Gupta**
"""

print("📝 Uploading README...")
with open("README_hf.md", "w", encoding="utf-8") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="README_hf.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model"
)
print("✅ README uploaded!")

# Create config
config = {
    "model_name": "Smart Waste Classification",
    "architecture": "MobileNetV2",
    "num_classes": 6,
    "classes": ["glass", "metal", "non-recyclable", "organic", "paper", "plastic"],
    "image_size": 224,
    "framework": "PyTorch",
    "pretrained_on": "ImageNet",
    "validation_accuracy": "97.46%",
    "input_normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}

print("📋 Uploading config...")
with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id=REPO_ID,
    repo_type="model"
)
print("✅ Config uploaded!")

# Upload training history if exists
import os
if os.path.exists("model/results/training_history.png"):
    print("📊 Uploading training plot...")
    api.upload_file(
        path_or_fileobj="model/results/training_history.png",
        path_in_repo="training_history.png",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✅ Training plot uploaded!")

# Cleanup
os.remove("README_hf.md")
os.remove("config.json")

print()
print("🎉 All done! Visit: https://huggingface.co/" + REPO_ID)
