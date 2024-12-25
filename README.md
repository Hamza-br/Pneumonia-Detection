# Pneumonia Detection using PyTorch and Tkinter

This repository provides a pipeline for training and deploying a pneumonia detection model using chest X-ray images. It includes preprocessing, model training, and a graphical user interface (GUI) for predictions.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Training the Model](#training-the-model)
5. [Using the GUI for Predictions](#using-the-gui-for-predictions)
6. [Acknowledgements](#acknowledgements)

---

## Overview

The project includes two main components:

1. **Training Pipeline**: A PyTorch-based pipeline for preprocessing chest X-ray images, training an enhanced pneumonia detection model using EfficientNet, and evaluating the model's performance.

2. **Graphical User Interface (GUI)**: A Tkinter-based application for loading the trained model and predicting pneumonia likelihood on chest X-ray images.

---

## Dataset

The dataset used for this project can be accessed from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data):

- **Images**: Chest X-ray images in DICOM format
- **CSV File**: Metadata including patient information and labels

Ensure to download the dataset and extract it locally. The folder structure should look like this:

```
pneumonio_detection/
|-- images/
|-- stage2_train_metadata.csv
```

---

## Setup

### Prerequisites

1. Install [Anaconda](https://www.anaconda.com/).
2. Download the dataset as described above.

### Environment Setup

Run the following commands to set up the environment:

```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Create and activate the conda environment
conda create --name pneumonia_env python=3.8 -y
conda activate pneumonia_env

# Install required dependencies
pip install -r requirements.txt
```

### Requirements

The required libraries are listed in the give file:

```
torch
torchvision
pandas
numpy
opencv-python
pillow
scikit-learn
tqdm
pydicom
tk
```

---

## Training the Model

1. Preprocess the dataset to convert DICOM images to PNG format:

   ```bash
   python preprocess.py --dataroot <path_to_dataset> --mode train
   python preprocess.py --dataroot <path_to_dataset> --mode test
   ```

2. Train the model:

   ```bash
   python train.py
   ```

   This script will split the dataset into training and validation sets, train the model, and save the best model as `enhanced_pneumonia_detection_model.pth`.

---

## Using the GUI for Predictions

1. Launch the GUI application:

   ```bash
   python gui.py
   ```

2. Use the GUI to:

   - Load the trained model file (`*.pth`).
   - Select a chest X-ray image for prediction.

3. The application will display the image and predict whether there is a high or low likelihood of pneumonia.

---

## Acknowledgements

This project uses data from the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) and leverages PyTorch and Tkinter for deep learning and GUI implementation.

