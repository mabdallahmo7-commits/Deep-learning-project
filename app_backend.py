from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import base64
import numpy as np
from io import BytesIO
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = YOLO('../model/best.pt')

@app.route('/')
def index():
    return send_from_directory(os.path.join(app.root_path, '..', 'frontend'), 'index.html')

@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Perform inference
        results = model(img)

        # Process results (assuming segmentation masks)
        # For simplicity, return the annotated image as base64
        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'segmented_image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Preprocess image (Histogram Equalization)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized_img = cv2.equalizeHist(gray_img)
        preprocessed_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)

        _, buffer = cv2.imencode('.jpg', preprocessed_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'preprocessed_image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluation', methods=['GET'])
def evaluation():
    try:
        # Path to the directory with evaluation results
        eval_dir = os.path.join(app.root_path, '..', 'training_runs', 'finetune_20251207_045229')

        result_img_path = os.path.join(eval_dir, 'results.png')
        confusion_matrix_path = os.path.join(eval_dir, 'confusion_matrix.png')

        if not os.path.exists(result_img_path) or not os.path.exists(confusion_matrix_path):
            return jsonify({'error': 'Evaluation files not found'}), 404

        with open(result_img_path, 'rb') as f:
            result_img_base64 = base64.b64encode(f.read()).decode('utf-8')

        with open(confusion_matrix_path, 'rb') as f:
            confusion_matrix_base64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            'result_image': result_img_base64,
            'confusion_matrix': confusion_matrix_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
