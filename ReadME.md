Hair Type Detector - CNN-based Image Classifier
## Overview
This project implements a Convolutional Neural Network (CNN) to classify images of different hair
types. The model classifies images into five categories: **Straight, Wavy, Curly, Dreadlocks, and
Kinky**. The dataset consists of grayscale images resized and augmented during training to improve
model performance and generalization.
## Table of Contents
- Overview
- Dataset
- Model Architecture
- Requirements
- Data Preprocessing
- Training the Model
- Evaluation
- Results
- Usage
- Contributing
## Dataset
The dataset https://www.kaggle.com/datasets/kavyasreeb/hair-type-dataset contains images of five hair types:
- Straight
- Wavy
- Curly
- Dreadlocks
- Kinky
Images are stored in different directories based on the hair type. The images are resized to 150x150
pixels, and grayscale conversion is applied before feeding them into the model.
## Model Architecture
The model is a CNN with the following structure:
- Input Layer: (150, 150, 1) for grayscale images.
- Convolutional Layers: Five convolutional layers with increasing filters (32, 64, 128, 256).
- MaxPooling Layers: Applied after each convolution to reduce dimensionality.
- Dropout Layers: To prevent overfitting.
- Dense Layers: Fully connected layers with ReLU activation.
- Output Layer: Softmax activation for multiclass classification (5 classes).
## Requirements
To run this project, you need the following libraries:
- Python 3.x
- TensorFlow
- Keras
- Numpy
- PIL (Pillow)
- scikit-learn
You can install the necessary packages using the following command:
```bash
pip install tensorflow keras numpy pillow scikit-learn
```
## Data Preprocessing
1. Images are resized to 150x150 pixels.
2. Images are converted to grayscale.
3. The dataset is split into training (80%) and testing (20%) sets.
4. Data augmentation is applied to the training set to increase robustness.
## Training the Model
The model is trained for 100 epochs with early stopping to prevent overfitting. The early stopping
callback monitors validation loss and stops training if there is no improvement for 10 epochs.
## Evaluation
After training, the model is evaluated on the test set to measure accuracy.
## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/hair_type_detector.git
cd hair_type_detector
```
2. Prepare the dataset:
- Place the images in the directory structure `data/Straight`, `data/Wavy`, `data/Curly`, etc.
3. Run the model:
```bash
python train_model.py
```
## Contributing
If you want to contribute to this project:
1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push the branch (`git push origin feature-branch`).
5. Open a pull request.