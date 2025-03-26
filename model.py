import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

class StressModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def build_model(self, input_shape, num_classes):
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
    
    def preprocess_training_data(self, X_train, y_train):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        noise_factor = 0.1
        X_train_noisy = X_train_resampled + np.random.normal(
            loc=0.0, scale=noise_factor, size=X_train_resampled.shape
        )
        
        X_train_combined = np.vstack((X_train_resampled, X_train_noisy))
        y_train_combined = np.hstack((y_train_resampled, y_train_resampled))
        
        y_train_onehot = tf.keras.utils.to_categorical(y_train_combined)
        
        return X_train_combined, y_train_onehot
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        X_processed, y_processed = self.preprocess_training_data(X_train, y_train)
        
        self.model = self.build_model(X_processed.shape[1], y_processed.shape[1])
        
        history = self.model.fit(
            X_processed, y_processed,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history
    
    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class] * 100
        
        return predicted_class, confidence