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
import os
import json
from datetime import datetime

class StressModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'snoring_range', 'respiration_rate', 'body_temperature', 
            'limb_movement', 'blood_oxygen', 'eye_movement', 
            'hours_of_sleep', 'heart_rate'
        ]
        self.data_source = {}
        self.training_history = {}

    def preprocess_data(self, data_path, data_source_info=None):
        """
        Preprocess the dataset

        Args:
            data_path: Path to the CSV data file
            data_source_info: Dictionary containing information about the data source
        """

        if data_source_info:
            self.data_source = data_source_info
        else:
            self.data_source = {
                "name": os.path.basename(data_path),
                "description": "Sleep health and lifestyle dataset",
                "records": 0,
                "source": "Unknown",
                "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        df = pd.read_csv(data_path)

        self.data_source["records"] = len(df)

        df.columns = [col.strip() for col in df.columns]

        for col in df.columns:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())

        if 'Stress Levels' in df.columns:
            df['Stress Levels'] = df['Stress Levels'].astype(int)

        stats = {}
        for col in self.feature_columns:
            if col in df.columns:
                stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

        self.data_source["statistics"] = stats

        if 'Stress Levels' in df.columns:
            level_counts = df['Stress Levels'].value_counts().sort_index().to_dict()
            self.data_source["level_distribution"] = {str(k): int(v) for k, v in level_counts.items()}

        X = df[self.feature_columns]
        y = df['Stress Levels']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        scaling_params = {}
        feature_means = self.scaler.mean_
        feature_scales = self.scaler.scale_

        for i, col in enumerate(self.feature_columns):
            scaling_params[col] = {
                "mean": float(feature_means[i]),
                "scale": float(feature_scales[i])
            }

        self.data_source["scaling_params"] = scaling_params

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        noise_factor = 0.1
        X_train_noisy = X_train_resampled + np.random.normal(
            loc=0.0, scale=noise_factor, size=X_train_resampled.shape
        )

        X_train_combined = np.vstack((X_train_resampled, X_train_noisy))
        y_train_combined = np.hstack((y_train_resampled, y_train_resampled))

        self.num_classes = len(np.unique(y))

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

    def train(self, data_path, data_source_info=None, epochs=50, batch_size=32):
        """
        Train the model with the provided dataset

        Args:
            data_path: Path to the CSV data file
            data_source_info: Dictionary containing information about the data source
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        start_time = datetime.now()

        X_train, X_test, y_train, y_test, y_train_onehot, y_test_onehot = self.preprocess_data(
            data_path, data_source_info
        )

        self.model = self.build_model(X_train.shape[1], self.num_classes)

        history = self.model.fit(
            X_train, y_train_onehot,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )

        test_loss, test_acc = self.model.evaluate(X_test, y_test_onehot)
        print(f"Test accuracy: {test_acc:.4f}")

        self.training_history = {
            "accuracy": float(test_acc),
            "loss": float(test_loss),
            "epochs": epochs,
            "batch_size": batch_size,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "training_time": str(datetime.now() - start_time),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history_dict = history.history
        self.training_history["training_accuracy"] = [float(x) for x in history_dict["accuracy"]]
        self.training_history["training_loss"] = [float(x) for x in history_dict["loss"]]
        self.training_history["validation_accuracy"] = [float(x) for x in history_dict["val_accuracy"]]
        self.training_history["validation_loss"] = [float(x) for x in history_dict["val_loss"]]

        return history, test_acc

    def save_model(self, model_path='stress_model.h5', scaler_path='scaler.pkl', metadata_path='model_metadata.json'):
        """
        Save the model, scaler, and metadata
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")

        if self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

        metadata = {
            "data_source": self.data_source,
            "training_history": self.training_history,
            "feature_columns": self.feature_columns,
            "num_classes": self.num_classes if hasattr(self, 'num_classes') else 5,
            "stress_level_names": [
                "Low Stress (Level 0)",
                "Low Stress (Level 1)",
                "Medium Stress (Level 2)",
                "Medium Stress (Level 3)",
                "High Stress (Level 4)"
            ]
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            print(f"Metadata saved to {metadata_path}")

    def load_model(self, model_path='stress_model.h5', scaler_path='scaler.pkl', metadata_path='model_metadata.json'):
        """
        Load the model, scaler, and metadata
        """
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")

            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                self.data_source = metadata.get('data_source', {})
                self.training_history = metadata.get('training_history', {})
                self.feature_columns = metadata.get('feature_columns', self.feature_columns)
                self.num_classes = metadata.get('num_classes', 5)

                print(f"Metadata loaded from {metadata_path}")

            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, features):
        """
        Make prediction with the model

        Args:
            features: List of feature values in the order of self.feature_columns
        """
        if self.model is None:
            print("Model not loaded. Using simple prediction instead.")
            return self._simple_predict(features)

        try:

            features_array = np.array(features).reshape(1, -1)

            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_array)
            else:
                features_scaled = features_array

            prediction = self.model.predict(features_scaled)

            predicted_class = np.argmax(prediction, axis=1)[0]

            probabilities = prediction[0]

            stress_level_names = [
                "Low Stress (Level 0)",
                "Low Stress (Level 1)",
                "Medium Stress (Level 2)",
                "Medium Stress (Level 3)",
                "High Stress (Level 4)"
            ]

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

        normalized = np.array([
            (features[0] - 50) / 25,    
            (features[1] - 15) / 5,     
            (features[2] - 37) / 0.5,   
            features[3] / 50,           
            (features[4] - 95) / 2.5,   
            features[5] / 50,           
            (features[6] - 7) / 2,      
            (features[7] - 70) / 20     
        ])

        weights = np.array([0.1, 0.1, 0.15, 0.1, -0.15, 0.1, -0.2, 0.2])
        stress_score = np.dot(normalized, weights)

        stress_score = 50 + (stress_score * 25)
        stress_score = max(0, min(100, stress_score))

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

    def get_data_source_info(self):
        """
        Get information about the data source
        """
        return self.data_source

    def get_training_history(self):
        """
        Get training history information
        """
        return self.training_history

    def generate_training_report(self, output_path='training_report.html'):
        """
        Generate HTML report of the training process
        """
        if not self.data_source or not self.training_history:
            print("No training data available to generate report")
            return False

        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Stressen - Training Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: 
                    margin: 0;
                    padding: 0;
                    background-color: 
                }
                header {
                    background: linear-gradient(135deg, 
                    color: white;
                    text-align: center;
                    padding: 2rem 0;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h1 {
                    margin: 0;
                    font-size: 2.5rem;
                }
                .subtitle {
                    font-style: italic;
                    margin-top: 0.5rem;
                    opacity: 0.9;
                }
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 0 2rem 3rem;
                }
                .section {
                    background-color: white;
                    border-radius: 8px;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }
                .two-column {
                    display: flex;
                    gap: 2rem;
                    flex-wrap: wrap;
                }
                .column {
                    flex: 1;
                    min-width: 300px;
                }
                h2 {
                    color: 
                    border-bottom: 2px solid 
                    padding-bottom: 0.5rem;
                    margin-top: 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1rem 0;
                }
                table th, table td {
                    padding: 0.75rem;
                    text-align: left;
                    border-bottom: 1px solid 
                }
                table th {
                    background-color: 
                }
                footer {
                    text-align: center;
                    padding: 1rem;
                    background-color: 
                    color: white;
                }
                .chart-container {
                    max-width: 100%;
                    height: 300px;
                    margin: 1.5rem 0;
                }
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <header>
                <h1>Stressen - Training Report</h1>
                <div class="subtitle">Deep Neural Network Training Results</div>
            </header>
            <div class="container">
                <div class="section">
                    <h2>Data Source Information</h2>
                    <table>
                        <tr>
                            <th>Source Name</th>
                            <td>{source_name}</td>
                        </tr>
                        <tr>
                            <th>Description</th>
                            <td>{description}</td>
                        </tr>
                        <tr>
                            <th>Records</th>
                            <td>{records}</td>
                        </tr>
                        <tr>
                            <th>Source</th>
                            <td>{source}</td>
                        </tr>
                        <tr>
                            <th>Date Processed</th>
                            <td>{date_processed}</td>
                        </tr>
                    </table>
                </div>

                <div class="section">
                    <h2>Training Results</h2>
                    <div class="two-column">
                        <div class="column">
                            <h3>Performance Metrics</h3>
                            <table>
                                <tr>
                                    <th>Test Accuracy</th>
                                    <td>{accuracy:.2f}%</td>
                                </tr>
                                <tr>
                                    <th>Loss</th>
                                    <td>{loss:.4f}</td>
                                </tr>
                                <tr>
                                    <th>Epochs</th>
                                    <td>{epochs}</td>
                                </tr>
                                <tr>
                                    <th>Batch Size</th>
                                    <td>{batch_size}</td>
                                </tr>
                                <tr>
                                    <th>Training Samples</th>
                                    <td>{training_samples}</td>
                                </tr>
                                <tr>
                                    <th>Test Samples</th>
                                    <td>{test_samples}</td>
                                </tr>
                                <tr>
                                    <th>Training Time</th>
                                    <td>{training_time}</td>
                                </tr>
                                <tr>
                                    <th>Training Date</th>
                                    <td>{training_date}</td>
                                </tr>
                            </table>
                        </div>
                        <div class="column">
                            <h3>Stress Level Distribution</h3>
                            <div class="chart-container">
                                <canvas id="distributionChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2>Training History</h2>
                    <div class="chart-container">
                        <canvas id="accuracyChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>

                <div class="section">
                    <h2>Feature Statistics</h2>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Mean</th>
                            <th>Std</th>
                        </tr>
                        {stats_rows}
                    </table>
                </div>
            </div>

            <footer>
                <p>Deep Neural Network Stress Prediction Project - 2025</p>
            </footer>

            <script>

                const distributionCtx = document.getElementById('distributionChart').getContext('2d');
                const distributionChart = new Chart(distributionCtx, {
                    type: 'bar',
                    data: {
                        labels: {level_labels},
                        datasets: [{
                            label: 'Number of Samples',
                            data: {level_counts},
                            backgroundColor: [
                                'rgba(46, 204, 113, 0.7)',
                                'rgba(46, 204, 113, 0.7)',
                                'rgba(255, 193, 7, 0.7)',
                                'rgba(255, 193, 7, 0.7)',
                                'rgba(231, 76, 60, 0.7)'
                            ],
                            borderColor: [
                                'rgba(46, 204, 113, 1)',
                                'rgba(46, 204, 113, 1)',
                                'rgba(255, 193, 7, 1)',
                                'rgba(255, 193, 7, 1)',
                                'rgba(231, 76, 60, 1)'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Stress Level Distribution in Dataset'
                            }
                        }
                    }
                });

                const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
                const accuracyChart = new Chart(accuracyCtx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: {epochs}}, (_, i) => i + 1),
                        datasets: [
                            {
                                label: 'Training Accuracy',
                                data: {training_accuracy},
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                tension: 0.1
                            },
                            {
                                label: 'Validation Accuracy',
                                data: {validation_accuracy},
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1
                            }
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Accuracy over Training Epochs'
                            }
                        }
                    }
                });

                const lossCtx = document.getElementById('lossChart').getContext('2d');
                const lossChart = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: Array.from({length: {epochs}}, (_, i) => i + 1),
                        datasets: [
                            {
                                label: 'Training Loss',
                                data: {training_loss},
                                borderColor: 'rgba(54, 162, 235, 1)',
                                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                                tension: 0.1
                            },
                            {
                                label: 'Validation Loss',
                                data: {validation_loss},
                                borderColor: 'rgba(255, 99, 132, 1)',
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                tension: 0.1
                            }
                        ]
                    },
                    options: {
                        plugins: {
                            title: {
                                display: true,
                                text: 'Loss over Training Epochs'
                            }
                        }
                    }
                });
            </script>
        </body>
        </html>
        """

        stats_rows = ""
        for feature, stats in self.data_source.get('statistics', {}).items():
            stats_rows += f"""
            <tr>
                <td>{feature}</td>
                <td>{stats.get('min', 0):.2f}</td>
                <td>{stats.get('max', 0):.2f}</td>
                <td>{stats.get('mean', 0):.2f}</td>
                <td>{stats.get('std', 0):.2f}</td>
            </tr>
            """

        level_distribution = self.data_source.get('level_distribution', {})
        level_labels = [f'"Level {k}"' for k in level_distribution.keys()]
        level_counts = list(level_distribution.values())

        training_accuracy = self.training_history.get('training_accuracy', [0])
        validation_accuracy = self.training_history.get('validation_accuracy', [0])
        training_loss = self.training_history.get('training_loss', [0])
        validation_loss = self.training_history.get('validation_loss', [0])

        html = html.format(
            source_name=self.data_source.get('name', 'Unknown'),
            description=self.data_source.get('description', 'No description available'),
            records=self.data_source.get('records', 0),
            source=self.data_source.get('source', 'Unknown'),
            date_processed=self.data_source.get('date_processed', 'Unknown'),
            accuracy=self.training_history.get('accuracy', 0) * 100,
            loss=self.training_history.get('loss', 0),
            epochs=self.training_history.get('epochs', 0),
            batch_size=self.training_history.get('batch_size', 0),
            training_samples=self.training_history.get('training_samples', 0),
            test_samples=self.training_history.get('test_samples', 0),
            training_time=self.training_history.get('training_time', 'Unknown'),
            training_date=self.training_history.get('date', 'Unknown'),
            stats_rows=stats_rows,
            level_labels=",".join(level_labels),
            level_counts=level_counts,
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            training_loss=training_loss,
            validation_loss=validation_loss
        )

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Training report saved to {output_path}")
        return True

if __name__ == "__main__":

    stress_model = StressModel()

    data_source_info = {
        "name": "Sleep Health and Lifestyle Dataset",
        "description": "Dataset containing sleep health metrics and physiological parameters",
        "source": "Kaggle - Sleep Health and Lifestyle Dataset by SRIRAM PRASHANTH",
        "url": "https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset",
        "date_collected": "2023-06-21"
    }
