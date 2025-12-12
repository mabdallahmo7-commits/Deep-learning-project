# Deep-learning-project
Road cracks instance segmentation 
# Road Crack Detection Project


1. Problem Definition
The main goal of this project is to automatically detect and segment cracks in road surfaces from images. This is a critical task for infrastructure maintenance, as it allows for timely repairs and prevents further deterioration of roads. By automating this process, we can make it faster, more efficient, and less subjective than manual inspections.

2. Dataset
The dataset used for this project is the "Crack Segmentation" dataset. It contains images of road surfaces with and without cracks, along with corresponding segmentation masks that delineate the crack regions.

The dataset is structured as follows:
- The main dataset directory contains :-
- train: Contains the training images and labels.
- valid: Contains the validation images and labels.
- test: Contains the test images and labels.

3. Preprocessing
Before feeding the images into the model, some preprocessing steps are performed. The frontend application includes a "Preprocess" feature that demonstrates one of these steps: Histogram Equalization. This technique enhances the contrast of the images, which can help in making the cracks more prominent and easier for the model to detect. The implementation for this can be found in the (backend_app.py) file.

4. Feature Extraction & Algorithm
The core of this project is a deep learning model for image segmentation. The chosen algorithm is YOLOv8, a state-of-the-art, real-time object detection and image segmentation model.

YOLOv8 is used to automatically extract relevant features from the input images and learn to distinguish between cracked and non-cracked road surfaces. The model architecture is defined and loaded using the ultralytics Python library.

5. Model Training
The model was trained on the "Crack Segmentation" dataset. The training process involves two main stages:

1.  Initial Training: A pre-trained YOLOv8 segmentation model yolov8n-seg.pt was trained on the custom dataset for 50 epochs. The script for this is train/train.py. The best performing model from this stage is saved as `model/best.pt`.
2.  Fine-Tuning: The model from the initial training was then fine-tuned for another 50 epochs to further improve its performance on the specific task of crack detection. The script for this is `train/train fine-tunning.py`.

The training runs, including logs, weights, and results, are saved in the `runs/` and `training_runs/` directories.

6. Evaluation
The model's performance is evaluated using various metrics, which are generated during the training and validation phases. The frontend application has an "Evaluation" feature that displays some of these results.
