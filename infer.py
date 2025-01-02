# infer.py

import sys
import os
import logging
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up logging
logger = logging.getLogger('infer_logger')
logger.setLevel(logging.DEBUG)

# Create handlers for console and file logging
c_handler = logging.StreamHandler(sys.stdout)
f_handler = logging.FileHandler('infer.log')

c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Constants
CLASSES = ['__background__', 'recycling', 'nonrecycling']  # Include background
NUM_CLASSES = len(CLASSES)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    logger.info("Initialized Faster R-CNN model with MobileNetV3 backbone.")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("Modified the model's box predictor to accommodate the new number of classes.")

    return model

def infer(image_path):
    logger.info(f"Starting inference on image: {image_path}")

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load model
    device = torch.device('cpu')
    model = get_model_instance_segmentation(NUM_CLASSES)
    try:
        model.load_state_dict(torch.load('fasterrcnn_model.pth', map_location=device))
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    model.eval()
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Image {image_path} loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        sys.exit(1)

    input_tensor = transform(image).to(device)
    logger.debug(f"Image transformed and tensor created with shape {input_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model([input_tensor])
    logger.debug("Model inference completed.")

    # Process predictions
    outputs = outputs[0]
    boxes = outputs['boxes']
    labels = outputs['labels']
    scores = outputs['scores']

    # Set a confidence threshold
    confidence_threshold = 0.5

    # Display detections
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    num_detections = 0
    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            num_detections += 1
            xmin, ymin, xmax, ymax = box
            class_name = CLASSES[label]
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin - 10, f'{class_name}: {score:.2f}', color='red', fontsize=12)
            logger.info(f"Detected object: {class_name}, Confidence {score:.2f}, "
                        f"Box [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")

    if num_detections == 0:
        logger.warning("No objects detected with confidence > 0.5.")
    else:
        logger.info(f"Total objects detected: {num_detections}")

    plt.show()
    logger.info("Inference completed and results displayed.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        logger.error("Usage: python infer.py <image_path>")
    else:
        infer(sys.argv[1])
