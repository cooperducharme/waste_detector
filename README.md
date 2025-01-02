
# Waste Sorting Model with Faster R-CNN

This repository contains two scripts, `train.py` and `infer.py`, for training and inference of a waste sorting model based on Faster R-CNN with MobileNetV3 backbone. This model can be fine-tuned to detect two classes: **recycling** and **nonrecycling** objects in images.

## Files in This Repository

1. **train.py**: This script trains a Faster R-CNN model using a custom dataset and logs training progress. The model is saved to `fasterrcnn_model.pth` for later use in inference.
2. **infer.py**: This script loads the trained model, performs inference on a provided image, and displays detected bounding boxes with class names and confidence scores.

## Requirements

Install the dependencies with:
```bash
pip install -r requirements.txt 
```

## Dataset Structure

Your dataset should be structured as follows:
```
dataset/
├── train/
│   ├── images/
│   ├── labels/
```

- **images/**: Contains `.jpg` images.
- **labels/**: Contains corresponding `.txt` files for bounding box annotations. Each `.txt` file should contain lines with:
  ```
  <class_id> <center_x> <center_y> <width> <height>
  ```
  All values are normalized between 0 and 1.

## Usage

### Training

To train the model, simply run:
```bash
python train.py
```
- The model will be saved as `fasterrcnn_model.pth` in the root directory.
- Logs will be saved to `train.log`.

### Inference

To perform inference on an image, use:
```bash
python infer.py <image_path>
```
- Replace `<image_path>` with the path to your image file.
- A window will display the image with detected objects, showing bounding boxes and confidence scores.
- Logs will be saved to `infer.log`.

### Classes

The model is trained to detect the following classes:
1. **recycling**
2. **nonrecycling**

Note: Background is considered as a separate class internally but is not labeled in the output.

## Scripts Overview

### train.py

- **Dataset Class**: The `WasteDataset` class loads images and their corresponding labels.
- **Transforms**: Images are transformed to tensors using `transforms.Compose([transforms.ToTensor()])`.
- **Model Initialization**: A pre-trained Faster R-CNN with a MobileNetV3 backbone is fine-tuned for our classes.
- **Training Loop**: The model is trained over epochs, and each epoch’s loss is logged to `train.log`.
- **Save Model**: The model is saved to `fasterrcnn_model.pth` after training completion.

### infer.py

- **Load Model**: The trained model is loaded from `fasterrcnn_model.pth`.
- **Image Processing**: Each image is transformed and fed to the model for inference.
- **Bounding Box Visualization**: Detected objects are visualized with bounding boxes, class names, and confidence scores.
- **Logging**: Each detection's details are logged to `infer.log`.

## Logging

Both `train.py` and `infer.py` scripts log messages to console and file:
- **train.log**: Logs details of the training process.
- **infer.log**: Logs details of the inference process, including any errors or detected objects.

## Model

The model is based on **Faster R-CNN with MobileNetV3** backbone, allowing lightweight and efficient training and inference on CPUs.

## Notes

- Adjust `train.py` hyperparameters as needed.
- Ensure dataset structure aligns with the specified format.

## Acknowledgments

This repository was developed as part of the **Waste Sorting Project** to efficiently classify recycling and non-recycling items.