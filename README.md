# Handwritten Math Operation Solver

A deep learning application that recognizes handwritten mathematical expressions and evaluates them.

## Prerequisites

- Python 3.8 or higher
- pip

## Setup and Run (3 Simple Steps)

### Step 1: Create Virtual Environment and Install Requirements

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
# source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python model_train.py
```

This will:
- Load training images from the `extracted_images` folder
- Train a CNN model to recognize digits (0-9) and operators (+, -, *)
- Save the trained model as `model_final.json` and `model_final.weights.h5`
- Generate `train_final.csv` with processed training data

**Note:** Training may take several minutes depending on your hardware.

### Step 3: Run the Application

```bash
python model_run.py
```

This will open a GUI where you can:
- **Draw** mathematical expressions directly on the canvas
- **Upload** an image file containing handwritten math
- Get predictions and evaluated results

## Features

- Recognizes handwritten digits: 0-9
- Recognizes operators: +, -, *
- Interactive drawing canvas
- Image file upload support
- Real-time prediction and evaluation

## Model Architecture

- Convolutional Neural Network (CNN)
- Input: 28x28 grayscale images
- Output: 13 classes (0-9, +, -, *)
- Training: 10 epochs with categorical crossentropy loss

## File Structure

```
├── extracted_images/       # Training dataset (organized by class)
├── model_train.py         # Model training script
├── model_run.py           # GUI application
├── requirements.txt       # Python dependencies
├── model_final.json       # Trained model architecture (generated)
├── model_final.weights.h5 # Trained model weights (generated)
└── train_final.csv        # Processed training data (generated)
```

## Troubleshooting

- If you encounter issues with TensorFlow, ensure you have the correct version for your Python installation
- Make sure the `extracted_images` folder contains the training data organized in subfolders by class
- On some systems, you may need to install tkinter separately (usually comes with Python)

