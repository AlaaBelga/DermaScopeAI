import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
from flask import Flask, request, jsonify
import traceback 
from flask_cors import CORS

# --- 1. Initialize the Flask App ---
app = Flask(__name__)
CORS(app)

# --- Define Image Preprocessing (used by both models) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. LOAD SCREENING MODEL (NON-CANCEROUS) ---
screening_model = models.efficientnet_b0(weights=None) 
num_screening_classes = 9 
num_ftrs_screening = screening_model.classifier[1].in_features
screening_model.classifier[1] = nn.Linear(num_ftrs_screening, num_screening_classes) 
screening_model.load_state_dict(torch.load('non_cancerous_skin_model.pth', map_location=torch.device('cpu')))
screening_model.eval()

screening_class_names = [
    'Atopic Eczema', 'Keratosis Pilaris', 'Ringworm', 'Unclassified2',
    'chickenpox', 'normal skin', 'psoriasis', 'random object', 'rosacea'
]

# --- 3. LOAD CANCER DETECTION MODEL ---
cancer_model = models.efficientnet_b0(weights=None)
num_cancer_classes = 4
num_ftrs_cancer = cancer_model.classifier[1].in_features
cancer_model.classifier[1] = nn.Linear(num_ftrs_cancer, num_cancer_classes)
cancer_model.load_state_dict(torch.load('dermatology_model_best_efficientnet.pth', map_location=torch.device('cpu')))
cancer_model.eval()

cancer_class_names = ['Melanoma', 'Basal-cell carcinoma', 'Nevus', 'Unclassified']

# --- 4. CREATE API ENDPOINTS ---

def analyze_image(image_bytes, model, class_names):
    """Helper function to run prediction on an image."""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    results = []
    for i, prob in enumerate(probabilities):
        results.append({
            'class_name': class_names[i],
            'probability': prob.item() * 100
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

@app.route('/predict/screening', methods=['POST'])
def predict_screening():
    """Endpoint for the public screening (non-cancerous) model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        image_bytes = file.read()
        results = analyze_image(image_bytes, screening_model, screening_class_names)
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict/cancer', methods=['POST'])
def predict_cancer():
    """Endpoint for the professional cancer detection model."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    try:
        image_bytes = file.read()
        results = analyze_image(image_bytes, cancer_model, cancer_class_names)
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)