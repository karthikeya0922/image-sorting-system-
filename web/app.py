"""
Flask API for Waste Classification
Serves predictions using the Hugging Face model
"""

import os
import io
import json
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
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

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model():
    """Load model from Hugging Face Hub"""
    global model
    
    print("📥 Downloading model from Hugging Face...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    
    print("🧠 Loading model...")
    # Import the model architecture from the project
    import sys
    sys.path.insert(0, '.')
    from model.model_architecture import create_model
    
    model = create_model(architecture='mobilenet_v2', num_classes=len(CLASSES), pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model


def predict_image(image):
    """Make prediction on an image"""
    global model
    
    if model is None:
        load_model()
    
    # Preprocess
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_score = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {CLASSES[i]: round(probabilities[0][i].item() * 100, 2) 
                 for i in range(len(CLASSES))}
    
    return {
        'class': predicted_class,
        'confidence': round(confidence_score, 2),
        'info': CLASS_INFO[predicted_class],
        'all_probabilities': all_probs
    }


@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': CLASSES
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    print("📸 Received prediction request...")
    image = None
    
    try:
        # 1. Check for file upload (standard way)
        if 'image' in request.files:
            print("📁 Found image in request.files")
            file = request.files['image']
            if file.filename != '':
                image = Image.open(file.stream)
        
        # 2. Check for base64 in JSON
        if image is None and request.is_json:
            print("📝 Checking for base64 in JSON body")
            data = request.get_json()
            if data and 'image_base64' in data:
                image_data = data['image_base64']
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
        
        # 3. Check for base64 in form data
        if image is None and 'image_base64' in request.form:
            print("📝 Found base64 in form data")
            image_data = request.form['image_base64']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

        if image is None:
            print("❌ No image found in request!")
            return jsonify({
                'success': False, 
                'error': 'No image found. Please provide an image file or base64 data.'
            }), 400
        
        # Make prediction
        print("🔮 Making prediction...")
        result = predict_image(image)
        print(f"🎯 Predicted: {result['class']} ({result['confidence']}%)")
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        print(f"🔥 Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes')
def get_classes():
    """Get available classes"""
    return jsonify({
        'classes': CLASSES,
        'info': CLASS_INFO
    })


if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run server
    print("\n🚀 Starting server...")
    print("📍 Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
