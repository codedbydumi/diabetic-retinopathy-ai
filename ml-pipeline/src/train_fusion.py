"""
Multi-Modal Fusion Model
Combines clinical and image models for final prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import json
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FusionModel:
    def __init__(self):
        self.data_path = Path("ml-pipeline/data")
        self.model_path = Path("ml-pipeline/models")
        
        # Load pre-trained models
        self.clinical_model = None
        self.image_model = None
        self.fusion_model = None
        
        self.results = {}
        
    def load_pretrained_models(self):
        """Load the pre-trained clinical and image models"""
        print("📦 Loading pre-trained models...")
        
        # Load clinical model (using ensemble)
        self.clinical_model = joblib.load(self.model_path / 'clinical_ensemble.pkl')
        print("   ✅ Clinical ensemble model loaded")
        
        # Load image model
        self.image_model = tf.keras.models.load_model(
            self.model_path / 'retina_image_model.h5'
        )
        print("   ✅ Image model loaded")
        
        # Load scaler for clinical data
        self.scaler = joblib.load(self.model_path / 'clinical_scaler.pkl')
        print("   ✅ Clinical scaler loaded")
    
    def prepare_fusion_data(self):
        """Prepare data for fusion model training"""
        print("\n📊 Preparing fusion data...")
        
        # Load clinical data
        train_clinical = pd.read_csv(self.data_path / 'train' / 'clinical_train.csv')
        val_clinical = pd.read_csv(self.data_path / 'val' / 'clinical_val.csv')
        test_clinical = pd.read_csv(self.data_path / 'test' / 'clinical_test.csv')
        
        # Load image metadata
        train_images = pd.read_csv(self.data_path / 'train' / 'image_metadata_train.csv')
        val_images = pd.read_csv(self.data_path / 'val' / 'image_metadata_val.csv')
        test_images = pd.read_csv(self.data_path / 'test' / 'image_metadata_test.csv')
        
        # For demonstration, we'll create synthetic predictions
        # In production, you'd get actual predictions from the models
        
        # Get clinical predictions
        X_train_clinical = self.scaler.transform(train_clinical.drop('outcome', axis=1))
        X_val_clinical = self.scaler.transform(val_clinical.drop('outcome', axis=1))
        X_test_clinical = self.scaler.transform(test_clinical.drop('outcome', axis=1))
        
        self.train_clinical_pred = self.clinical_model.predict_proba(X_train_clinical)[:, 1]
        self.val_clinical_pred = self.clinical_model.predict_proba(X_val_clinical)[:, 1]
        self.test_clinical_pred = self.clinical_model.predict_proba(X_test_clinical)[:, 1]
        
        # Simulate image predictions (in production, use actual model predictions)
        # Convert DR grades to diabetes probability
        self.train_image_pred = self.grade_to_diabetes_prob(train_images['dr_grade'].values)
        self.val_image_pred = self.grade_to_diabetes_prob(val_images['dr_grade'].values)
        self.test_image_pred = self.grade_to_diabetes_prob(test_images['dr_grade'].values)
        
        # True labels
        self.y_train = train_clinical['outcome'].values
        self.y_val = val_clinical['outcome'].values
        self.y_test = test_clinical['outcome'].values
        
        # Ensure same length (some patients might not have images)
        min_len_train = min(len(self.train_clinical_pred), len(self.train_image_pred))
        min_len_val = min(len(self.val_clinical_pred), len(self.val_image_pred))
        min_len_test = min(len(self.test_clinical_pred), len(self.test_image_pred))
        
        self.train_clinical_pred = self.train_clinical_pred[:min_len_train]
        self.train_image_pred = self.train_image_pred[:min_len_train]
        self.y_train = self.y_train[:min_len_train]
        
        self.val_clinical_pred = self.val_clinical_pred[:min_len_val]
        self.val_image_pred = self.val_image_pred[:min_len_val]
        self.y_val = self.y_val[:min_len_val]
        
        self.test_clinical_pred = self.test_clinical_pred[:min_len_test]
        self.test_image_pred = self.test_image_pred[:min_len_test]
        self.y_test = self.y_test[:min_len_test]
        
        print(f"   ✅ Train samples: {len(self.y_train)}")
        print(f"   ✅ Validation samples: {len(self.y_val)}")
        print(f"   ✅ Test samples: {len(self.y_test)}")
    
    def grade_to_diabetes_prob(self, grades):
        """Convert DR grade to diabetes probability"""
        # Higher DR grades correlate with diabetes
        prob_map = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 0.95}
        return np.array([prob_map[g] + np.random.normal(0, 0.05) for g in grades]).clip(0, 1)
    
    def build_neural_fusion_model(self):
        """Build a neural network for fusion"""
        print("\n🏗️ Building neural fusion model...")
        
        # Input layers
        clinical_input = keras.Input(shape=(1,), name='clinical_prob')
        image_input = keras.Input(shape=(1,), name='image_prob')
        
        # Combine inputs
        combined = layers.concatenate([clinical_input, image_input])
        
        # Hidden layers
        x = layers.Dense(16, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(1, activation='sigmoid', name='diabetes_prob')(x)
        
        # Create model
        self.fusion_model = models.Model(
            inputs=[clinical_input, image_input],
            outputs=output
        )
        
        # Compile
        self.fusion_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print("   ✅ Neural fusion model built")
        print(f"   Parameters: {self.fusion_model.count_params():,}")
    
    def train_fusion_model(self, epochs=50):
        """Train the fusion model"""
        print(f"\n🚀 Training fusion model for {epochs} epochs...")
        
        # Prepare data
        X_train = [self.train_clinical_pred.reshape(-1, 1), 
                   self.train_image_pred.reshape(-1, 1)]
        X_val = [self.val_clinical_pred.reshape(-1, 1), 
                 self.val_image_pred.reshape(-1, 1)]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        history = self.fusion_model.fit(
            X_train, self.y_train,
            validation_data=(X_val, self.y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✅ Fusion model trained")
        
        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Fusion Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Fusion Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('docs/fusion_model_training.png')
        plt.show()
    
    def evaluate_all_approaches(self):
        """Compare all fusion approaches"""
        print("\n📊 Evaluating fusion approaches...")
        
        results = {}
        
        # 1. Clinical only
        clinical_pred = (self.test_clinical_pred > 0.5).astype(int)
        results['Clinical Only'] = {
            'accuracy': accuracy_score(self.y_test, clinical_pred),
            'auc': roc_auc_score(self.y_test, self.test_clinical_pred)
        }
        
        # 2. Image only
        image_pred = (self.test_image_pred > 0.5).astype(int)
        results['Image Only'] = {
            'accuracy': accuracy_score(self.y_test, image_pred),
            'auc': roc_auc_score(self.y_test, self.test_image_pred)
        }
        
        # 3. Simple average
        avg_prob = (self.test_clinical_pred + self.test_image_pred) / 2
        avg_pred = (avg_prob > 0.5).astype(int)
        results['Simple Average'] = {
            'accuracy': accuracy_score(self.y_test, avg_pred),
            'auc': roc_auc_score(self.y_test, avg_prob)
        }
        
        # 4. Weighted average (clinical has more weight)
        weighted_prob = 0.6 * self.test_clinical_pred + 0.4 * self.test_image_pred
        weighted_pred = (weighted_prob > 0.5).astype(int)
        results['Weighted Average'] = {
            'accuracy': accuracy_score(self.y_test, weighted_pred),
            'auc': roc_auc_score(self.y_test, weighted_prob)
        }
        
        # 5. Neural fusion
        X_test = [self.test_clinical_pred.reshape(-1, 1), 
                  self.test_image_pred.reshape(-1, 1)]
        neural_prob = self.fusion_model.predict(X_test).flatten()
        neural_pred = (neural_prob > 0.5).astype(int)
        results['Neural Fusion'] = {
            'accuracy': accuracy_score(self.y_test, neural_pred),
            'auc': roc_auc_score(self.y_test, neural_prob)
        }
        
        # 6. Maximum probability
        max_prob = np.maximum(self.test_clinical_pred, self.test_image_pred)
        max_pred = (max_prob > 0.5).astype(int)
        results['Maximum'] = {
            'accuracy': accuracy_score(self.y_test, max_pred),
            'auc': roc_auc_score(self.y_test, max_prob)
        }
        
        # Display results
        print("\n" + "="*60)
        print("FUSION APPROACH COMPARISON")
        print("="*60)
        
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        for approach, metrics in results_df.iterrows():
            print(f"\n{approach}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   AUC-ROC:  {metrics['auc']:.4f}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy comparison
        approaches = list(results_df.index)
        accuracies = results_df['accuracy'].values
        
        bars1 = axes[0].bar(range(len(