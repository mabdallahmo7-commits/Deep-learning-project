
 Road Crack Segmentation Project

1. Problem Definition and Data Collection

**Problem Definition:**
The primary goal of this project is to automatically detect and segment cracks on road surfaces from images. This is a semantic segmentation task, where each pixel in an image is classified as either "crack" or "background".

**Data Collection:**
The dataset for this project is located in the `data/` directory, which is organized into `train/`, `valid/`, and `test/` sets. Each set contains:
- `images/`: The raw road images (in .jpg format).
- `labels/`: Annotations in YOLOv8 segmentation format (.txt files with polygon coordinates).
- `masks/`: The corresponding ground truth segmentation masks (.png images), where white pixels represent cracks and black pixels are the background.

The `yolo_to_masks.py` script is used to convert the YOLO-style labels into the binary mask images required for training the U-Net model.

------------

2. Feature Engineering

For this deep learning-based computer vision task, traditional feature engineering is minimal as the model learns the relevant features directly from the image pixels. The primary "feature engineering" steps are the preprocessing transformations applied to the images before they are fed into the model:

-   **Resizing:** All images are resized to a consistent dimension (e.g., 256x256 or 640x640 pixels).
-   **Normalization:** Image pixel values are normalized to a specific range (e.g., [0, 1]).
-   **Histogram Equalization:** An optional preprocessing step available in the web interface to enhance contrast.

These transformations are handled by the `torchvision.transforms` library in `design_model.py` and `backend.py`.

------------

3. Model Design

The core of this project is a U-Net deep learning model, which is a convolutional neural network architecture designed for fast and precise image segmentation. The model is implemented in PyTorch and its architecture is defined in `design_model.py` and `backend.py`.

The key components of the U-Net are:
-   **Encoder (Contracting Path):** A series of convolutional and max-pooling layers that capture the context in the image. It extracts high-level features from the input.
-   **Bottleneck:** The layer between the encoder and decoder that captures the most abstract features.
-   **Decoder (Expansive Path):** A series of up-sampling and convolutional layers that enables precise localization. It combines the high-level features from the encoder with up-sampled feature maps to reconstruct the segmentation map.
-   **Skip Connections:** Connections that concatenate feature maps from the encoder to the decoder, which helps the decoder recover fine-grained details lost during down-sampling.

-------------

4. Model Training

The model training process is defined and executed by the `design_model.py` script.

To train the model, run:
```bash
python design_model.py
```

This script will:
1.  Load the training and validation data from the `data/` directory using a custom `CrackDataset` class.
2.  Initialize the U-Net model.
3.  Define the loss function (a combination of Dice Loss and Binary Cross-Entropy) and the optimizer (Adam).
4.  Iterate through the specified number of epochs, training the model on the training data and evaluating it on the validation data.
5.  Save the trained model's weights to `model.pth`.

-------------

5. Model Testing and Inference

**Model Testing:**
The `test_model.py` script is used to evaluate the performance of the trained model on the test dataset.

To run the tests:
```bash
python test_model.py
```
This script loads the `model.pth` weights, runs inference on the images in `data/test/images`, and compares the predicted masks with the ground truth masks in `data/test/masks`. It calculates the Dice Coefficient as the primary performance metric and saves the visual predictions in the `test_masks/` folder.

**Inference:**
Real-time inference is handled by the `backend.py` script, which serves the trained model through a REST API.

--------------

6. GUI Implementation and Application Running

The project includes a simple web-based Graphical User Interface (GUI) for easy interaction with the model.

**GUI Implementation:**
The frontend is a single `index.html` file. It provides an interface for:
-   Uploading an image of a road.
-   Sending the image to the backend for segmentation.
-   Displaying the original image and the resulting segmentation mask side-by-side.

**Application Running:**
To run the full application:
1.  **Start the backend server:**
    ```bash
    python backend.py
    ```
    This will start a Flask server, typically on `http://127.0.0.1:5000`.

2.  **Access the GUI:**
    Open a web browser and navigate to `http://127.0.0.1:5000`. The `index.html` page will be served, and you can begin using the road crack detection system.
