import os
import io
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
import base64

# Import model architecture
import sys
sys.path.insert(0, '.')
from model.model_architecture import create_model

# Constants
REPO_ID = "karthikeya09/smart_image_recognation"
MODEL_FILE = "best_model.pth"
CLASSES = ['glass', 'metal', 'non-recyclable', 'organic', 'paper', 'plastic']
CLASS_INFO = {
    'glass': {'emoji': '💚', 'color': '#28a745', 'description': 'Glass bottles, jars'},
    'metal': {'emoji': '🔘', 'color': '#6c757d', 'description': 'Cans, foil, batteries'},
    'non-recyclable': {'emoji': '⚫', 'color': '#343a40', 'description': 'Mixed/contaminated waste'},
    'organic': {'emoji': '🟢', 'color': '#20c997', 'description': 'Food waste, plant matter'},
    'paper': {'emoji': '📄', 'color': '#8B4513', 'description': 'Newspapers, cardboard'},
    'plastic': {'emoji': '🔵', 'color': '#007bff', 'description': 'Bottles, bags, containers'}
}

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📥 Downloading model from {REPO_ID}...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

print("🧠 Loading model architecture...")
model = create_model(architecture='mobilenet_v2', num_classes=len(CLASSES), pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(img):
    if img is None:
        return {"error": "No image provided"}
    
    # Preprocess
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item() * 100
    
    all_probs = {CLASSES[i]: round(probabilities[0][i].item() * 100, 2) 
                 for i in range(len(CLASSES))}
    
    return {
        'class': predicted_class,
        'confidence': round(confidence_score, 2),
        'info': CLASS_INFO[predicted_class],
        'all_probabilities': all_probs
    }

# Read the HTML template
with open('web/templates/index.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Modify the HTML to work with Gradio's direct API if needed, 
# but for now we'll just use a simple Gradio interface as a backup 
# and serve the HTML as the primary view.

def gradio_predict(img):
    res = predict(img)
    return {CLASSES[i]: res['all_probabilities'][CLASSES[i]]/100 for i in range(len(CLASSES))}

# Create Gradio Interface
with gr.Blocks(title="Smart Waste Classifier") as demo:
    gr.HTML(html_content)
    
    # Hidden components for the API calls from the custom HTML if we wanted to bridge them,
    # but since this is for a Space, users can also use the default Gradio UI below it
    # or we can hide the Gradio UI and just use the HTML.
    
    with gr.Accordion("Standard Interface", open=False):
        image_input = gr.Image(type="pil", label="Upload Image")
        label_output = gr.Label(num_top_classes=6, label="Classification Results")
        image_input.change(gradio_predict, inputs=image_input, outputs=label_output)

# Launch
if __name__ == "__main__":
    demo.launch()
