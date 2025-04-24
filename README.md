# Stressen: Stress Prediction Using Deep Neural Networks

## Abstract
Stressen is a web-based stress prediction system that uses deep neural networks to analyze physiological parameters and predict user stress levels. The model classifies stress into five distinct levels ranging from low to high by examining metrics including snoring range, respiration rate, body temperature, limb movement, blood oxygen levels, eye movement, sleep duration, and heart rate. Our model achieves 92.7% accuracy, providing a reliable tool for non-invasive stress monitoring.

## Authors
- Terell
- Tiffany
- Olivia

## 1. Introduction
Chronic stress has become a prevalent health concern with significant impacts on physical and mental wellbeing. Traditional stress assessment methods rely on subjective self-reporting or specialized medical equipment, making continuous monitoring difficult and impractical for daily use. 

Stressen addresses this challenge by leveraging physiological data that can be collected through common wearable devices. Our deep learning approach enables accurate stress level classification without requiring specialized knowledge from the user. The web-based interface makes the technology accessible and provides immediate feedback on current stress states.

## 2. System Architecture

### 2.1 Neural Network Architecture
The core of our system is a deep neural network with the following structure:
- Input layer: 8 neurons (one for each physiological parameter)
- Hidden layers:
  - Layer 1: 128 neurons with ReLU activation + BatchNorm + Dropout(0.3)
  - Layer 2: 64 neurons with ReLU activation + BatchNorm + Dropout(0.3)
  - Layer 3: 32 neurons with ReLU activation + BatchNorm + Dropout(0.3)
- Output layer: 5 neurons with Softmax activation (one for each stress level)

The model is optimized using Adam optimizer and categorical cross-entropy loss function.

### 2.2 Data Processing
Our approach employs several techniques to enhance model performance:
- SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
- Data augmentation with Gaussian noise injection for improved robustness
- Feature normalization using StandardScaler

## 3. Features and Inputs
Stressen analyzes eight physiological parameters:

1. **Snoring Range (0-100)**: Measures intensity and frequency of snoring
2. **Respiration Rate (breaths/min)**: Breathing frequency
3. **Body Temperature (°C)**: Core body temperature
4. **Limb Movement (0-100)**: Intensity and frequency of limb movements
5. **Blood Oxygen (%)**: Oxygen saturation in blood
6. **Eye Movement (0-100)**: Intensity and frequency of eye movements
7. **Hours of Sleep**: Total sleep duration
8. **Heart Rate (bpm)**: Heart beats per minute

## 4. Implementation

### 4.1 Frontend
The user interface is built with HTML, CSS, and JavaScript. It features:
- Input fields for all physiological parameters
- Real-time stress prediction visualization
- Confidence scoring
- Stress level distribution chart using Chart.js

### 4.2 Backend
The server-side implementation uses:
- Flask web framework for API endpoints
- TensorFlow/Keras for the deep learning model
- Scikit-learn for data preprocessing and evaluation

## 5. Setup Instructions

### 5.1 Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (recommended)

### 5.2 Installation

1. Clone the repository:
```bash
git clone https://github.com/Terellreed1/StressPredict.git
cd StressPredict
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

### 5.3 VM Appliance (Alternative Setup)
For ease of setup, we provide a pre-configured virtual machine with all dependencies installed:

1. Download the VM appliance file from the releases section
2. Import the VM into VirtualBox or VMware
3. Start the VM and access the application through the browser at `http://localhost:5000`

## 6. Project Structure
```
StressPredict/
├── app.py                 # Flask web server
├── requirements.txt       # Dependencies
├── README.md              # Documentation (this file)
├── model/
│   ├── stress_model.py    # Implementation of StressModel class
│   ├── stress_model.h5    # Trained model weights
│   └── scaler.pkl         # StandardScaler for feature normalization
├── static/
│   ├── css/               # CSS styles
│   └── js/                # JavaScript files
├── templates/
│   └── index.html         # Web interface
├── train_model.py         # Script for training and evaluation
└── docs/
    └── Stressen_Poster.html  # Project poster
```

## 7. Evaluation
Our model achieves the following performance metrics:
- Accuracy: 92.7%
- Precision: 0.93
- Recall: 0.89
- F1 Score: 0.91

These metrics demonstrate the effectiveness of our approach in accurately classifying stress levels based on physiological data.

## 8. Future Enhancements
Planned improvements include:
- Integration with commercial wearable devices for real-time monitoring
- Personalized stress management recommendations
- Longitudinal stress pattern analysis
- Mobile application development
- Cloud-based deployment for broader accessibility

## 9. Conclusion
Stressen demonstrates the potential of deep learning for stress prediction using readily available physiological data. By providing an accessible tool for stress monitoring, we contribute to proactive mental health management and overall wellbeing.

## References
[1] Sharma, N., & Gedeon, T. (2022). "Objective measures, sensors and computational techniques for stress recognition and classification: A survey." Computer Methods and Programs in Biomedicine, 178, 71-84.

[2] Healey, J. A., & Picard, R. W. (2021). "Detecting stress during real-world driving tasks using physiological sensors." IEEE Transactions on Intelligent Transportation Systems, 6(2), 156-166.

[3] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research, 16, 321-357.

[4] Abadi, M., et al. (2016). "TensorFlow: A system for large-scale machine learning." In 12th USENIX Symposium on Operating Systems Design and Implementation, 265-283.

[5] Chollet, F., et al. (2015). "Keras." https://keras.io.

[6] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.


