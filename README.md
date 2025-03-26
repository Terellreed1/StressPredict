# Stress Prediction Using Deep Neural Networks

A machine learning system for predicting stress levels based on physiological data using deep neural networks.



## Overview

This project implements a deep neural network model that analyzes physiological parameters to classify stress levels. The system provides real-time predictions with visualization tools for easy interpretation of results.

## Features

- **Multi-class Stress Level Classification**: Predicts stress across multiple categories from low to high
- **Comprehensive Data Analysis**: Processes 8 different physiological parameters
- **Interactive Web Interface**: User-friendly input and visualization of results
- **High Accuracy**: Achieves 92.7% accuracy in stress level prediction
- **Real-time Feedback**: Immediate results with confidence metrics

## Technical Implementation

### Model Architecture

The system uses a deep neural network with:
- Input layer for physiological parameters
- Three hidden layers with ReLU activation, batch normalization, and dropout
- Softmax output layer for multi-class classification

### Data Preprocessing

Advanced preprocessing techniques implemented:
- Standardization using z-score normalization
- SMOTE for handling class imbalance
- Data augmentation with Gaussian noise
- One-hot encoding for categorical target variables

### Tech Stack

- **Frontend**: HTML, CSS, JavaScript with Chart.js
- **Backend**: Python Flask API
- **ML Framework**: TensorFlow/Keras
- **Data Processing**: scikit-learn, imbalanced-learn, pandas, numpy

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- TensorFlow 2.8+
- scikit-learn
- imbalanced-learn
- Other dependencies in requirements.txt

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stress-prediction.git
   cd stress-prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```
   python app.py
   ```

4. Open the interface:
   - Open `index.html` with VSCode Live Server or simply by opening it in a browser

## Usage

1. Input the required physiological parameters:
   - Snoring range
   - Respiration rate
   - Body temperature
   - Limb movement
   - Blood oxygen
   - Eye movement
   - Hours of sleep
   - Heart rate

2. Click "Predict Stress Level" to get results
3. View the prediction along with confidence metrics and distribution visualization

## Dataset

The model is trained on a dataset containing 630+ records with physiological parameters and corresponding stress levels. Data augmentation techniques were used to expand the training set.

## Future Development

- Questionnaire-based prediction for users without physiological monitoring devices
- Integration with wearable devices for continuous monitoring
- User accounts for tracking stress levels over time
- Mobile application for on-the-go stress monitoring



