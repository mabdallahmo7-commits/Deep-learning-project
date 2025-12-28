from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import os

app = Flask(__name__)
CORS(app)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024, 512)
        self.dec3 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec1 = self.conv_block(128, 64)

        # Final output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # Separable Conv 1
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Separable Conv 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        file_data = file.read()
        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        original_size = img.shape[:2]  # (height, width)

        # Convert to PIL for transform
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Preprocess
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor))
            print('Model output sum:', output.sum().item())  # Debug: print sum of output mask
            pred = (output > 0.3).float().squeeze().cpu().numpy()  # Lower threshold to 0.3

        # Resize back
        pred_resized = cv2.resize(pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

        # Convert to image
        pred_img = (pred_resized * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', pred_img)
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

if __name__ == '__main__':
    app.run(debug=True)
