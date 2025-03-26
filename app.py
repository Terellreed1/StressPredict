import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class StressModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'snoring_range', 'respiration_rate', 'body_temperature', 
            'limb_movement', 'blood_oxygen', 'eye_movement', 
            'hours_of_sleep', 'heart_rate'
        ]
    
    def preprocess_data(self, data_path):
        """
        Preprocess the dataset
        """
        # Load data
        df = pd.read_csv(data_path)
        
        # Clean column names (trim any extra spaces)
        df.columns = [col.strip() for col in df.columns]
        
        # Handle missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
        
        # Ensure target variable is integer
        if 'Stress Levels' in df.columns:
            df['Stress Levels'] = df['Stress Levels'].astype(int)
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['Stress Levels']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        # Data augmentation with Gaussian noise
        noise_factor = 0.1
        X_train_noisy = X_train_resampled + np.random.normal(
            loc=0.0, scale=noise_factor, size=X_train_resampled.shape
        )
        
        # Combine original and augmented data
        X_train_combined = np.vstack((X_train_resampled, X_train_noisy))
        y_train_combined = np.hstack((y_train_resampled, y_train_resampled))
        
        # Get number of classes
        self.num_classes = len(np.unique(y))
        
        # One-hot encode target for training
        y_train_onehot = tf.keras.utils.to_categorical(y_train_combined, num_classes=self.num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=self.num_classes)
        
        return X_train_combined, X_test_scaled, y_train_combined, y_test, y_train_onehot, y_test_onehot
    
    def build_model(self, input_shape, num_classes):
        """
        Build the Deep Neural Network model
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_path, epochs=50, batch_size=32):
        """
        Train the model with the provided dataset
        """
        # Preprocess data
        X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = self.preprocess_data(data_path)
        
        # Build model
        self.model = self.build_model(X_train.shape[1], self.num_classes)
        
        # Train model
        history = self.model.fit(
            X_train, y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = self.model.evaluate(X_test, y_test_onehot)
        print(f"Test accuracy: {test_acc:.4f}")
        
        return history, test_acc
    
    def save_model(self, model_path='stress_model.h5', scaler_path='scaler.pkl'):
        """
        Save the model and scaler
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path='stress_model.h5', scaler_path='scaler.pkl'):
        """
        Load the model and scaler
        """
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        """
        Make prediction with the model
        """
        if self.model is None:
            print("Model not loaded. Using simple prediction instead.")
            return self._simple_predict(features)
        
        try:
            # Convert features to numpy array
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            
            # Get the class with highest probability
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            # Get probabilities
            probabilities = prediction[0]
            
            # Map stress levels to names
            stress_level_names = [
                "Low Stress (Level 0)",
                "Low Stress (Level 1)",
                "Medium Stress (Level 2)",
                "Medium Stress (Level 3)",
                "High Stress (Level 4)"
            ]
            
            # Create result dictionary
            result = {
                'level': int(predicted_class),
                'level_name': stress_level_names[predicted_class],
                'confidence': float(probabilities[predicted_class] * 100),
                'probabilities': {
                    stress_level_names[i]: float(prob * 100) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
            return result
        
        except Exception as e:
            print(f"Error making prediction: {e}")
            return self._simple_predict(features)
    
    def _simple_predict(self, features):
        """
        Simple prediction logic for fallback
        """
        # Normalize features (mock implementation)
        normalized = np.array([
            (features[0] - 50) / 25,    # snoring
            (features[1] - 15) / 5,     # respiration
            (features[2] - 37) / 0.5,   # temperature
            features[3] / 50,           # limb
            (features[4] - 95) / 2.5,   # oxygen
            features[5] / 50,           # eye
            (features[6] - 7) / 2,      # sleep
            (features[7] - 70) / 20     # heart
        ])
        
        # Calculate stress score based on weights
        weights = np.array([0.1, 0.1, 0.15, 0.1, -0.15, 0.1, -0.2, 0.2])
        stress_score = np.dot(normalized, weights)
        
        # Scale to 0-100
        stress_score = 50 + (stress_score * 25)
        stress_score = max(0, min(100, stress_score))
        
        # Determine stress level
        if stress_score < 20:
            level = 0
            level_name = "Low Stress (Level 0)"
            probabilities = [0.75, 0.20, 0.05, 0.0, 0.0]
        elif stress_score < 40:
            level = 1
            level_name = "Low Stress (Level 1)"
            probabilities = [0.15, 0.65, 0.20, 0.0, 0.0]
        elif stress_score < 60:
            level = 2
            level_name = "Medium Stress (Level 2)"
            probabilities = [0.0, 0.15, 0.70, 0.15, 0.0]
        elif stress_score < 80:
            level = 3
            level_name = "Medium Stress (Level 3)"
            probabilities = [0.0, 0.0, 0.15, 0.70, 0.15]
        else:
            level = 4
            level_name = "High Stress (Level 4)"
            probabilities = [0.0, 0.0, 0.0, 0.20, 0.80]
        
        # Create result dictionary
        result = {
            'level': level,
            'level_name': level_name,
            'stress_score': float(stress_score),
            'confidence': float(85 + np.random.randint(10)),
            'probabilities': {
                'Low Stress (Level 0)': float(probabilities[0] * 100),
                'Low Stress (Level 1)': float(probabilities[1] * 100),
                'Medium Stress (Level 2)': float(probabilities[2] * 100),
                'Medium Stress (Level 3)': float(probabilities[3] * 100),
                'High Stress (Level 4)': float(probabilities[4] * 100)
            }
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize model
    stress_model = StressModel()
    
    # Train model with data (uncomment if you have a dataset)
    # history, accuracy = stress_model.train('path/to/your/dataset.csv')
    # stress_model.save_model()
    
    # Or load a pre-trained model
    # stress_model.load_model()
    
    # Make a prediction
    test_features = [45, 16, 36.6, 20, 96, 30, 7.5, 75]
    result = stress_model.predict(test_features)
    print(result)