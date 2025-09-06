"""
Image Model Training Script
Train CNN model for diabetic retinopathy detection using transfer learning
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImageModelTrainer:
    def __init__(self):
        self.data_path = Path("ml-pipeline/data")
        self.model_path = Path("ml-pipeline/models")
        self.model_path.mkdir(exist_ok=True)
        
        # Image parameters
        self.img_size = (224, 224)  # EfficientNet input size
        self.batch_size = 32
        self.num_classes = 5  # DR grades 0-4
        
        self.model = None
        self.history = None
        self.results = {}
        
    def create_synthetic_images(self):
        """
        Create synthetic images for training
        In production, you'd load real retinal images
        """
        print("🖼️ Creating synthetic training images...")
        
        # Load metadata
        metadata_files = {
            'train': self.data_path / 'train' / 'image_metadata_train.csv',
            'val': self.data_path / 'val' / 'image_metadata_val.csv',
            'test': self.data_path / 'test' / 'image_metadata_test.csv'
        }
        
        for split, metadata_file in metadata_files.items():
            df = pd.read_csv(metadata_file)
            split_dir = self.data_path / split / 'images'
            split_dir.mkdir(exist_ok=True)
            
            # Create synthetic images for each grade
            for _, row in df.iterrows():
                grade = row['dr_grade']
                grade_dir = split_dir / f'grade_{grade}'
                grade_dir.mkdir(exist_ok=True)
                
                # Create a synthetic image (in production, load real image)
                # For now, create a placeholder image with different patterns per grade
                img = self.create_synthetic_retina_image(grade)
                
                # Save image
                img_path = grade_dir / f"{row['image_id']}.png"
                tf.keras.preprocessing.image.save_img(img_path, img)
            
            print(f"   ✅ Created {len(df)} images for {split} set")
    
    def create_synthetic_retina_image(self, grade):
        """
        Create a synthetic retinal image based on DR grade
        In production, this would be replaced with real image loading
        """
        # Create base circular pattern resembling retina
        img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        center = (self.img_size[0] // 2, self.img_size[1] // 2)
        
        # Base retina color (reddish)
        base_color = np.array([180 - grade * 20, 80 - grade * 10, 60])
        
        # Create circular gradient
        for i in range(self.img_size[0]):
            for j in range(self.img_size[1]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                if dist < 100:
                    intensity = 1.0 - (dist / 100)
                    img[i, j] = base_color * intensity
        
        # Add noise to simulate vessels and features
        noise = np.random.normal(0, 10 + grade * 5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add synthetic features based on grade
        if grade > 0:
            # Simulate microaneurysms (small dots)
            num_spots = grade * 3
            for _ in range(num_spots):
                x, y = np.random.randint(20, self.img_size[0]-20, 2)
                img[x-2:x+2, y-2:y+2] = [200, 100, 100]
        
        if grade > 2:
            # Simulate hemorrhages (larger spots)
            num_hemorrhages = grade - 2
            for _ in range(num_hemorrhages):
                x, y = np.random.randint(30, self.img_size[0]-30, 2)
                img[x-5:x+5, y-5:y+5] = [150, 50, 50]
        
        return img
    
    def prepare_data_generators(self):
        """Prepare data generators for training"""
        print("\n📊 Preparing data generators...")
        
        # Create synthetic images first
        self.create_synthetic_images()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            fill_mode='constant',
            cval=0
        )
        
        # Only rescaling for validation/test
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path / 'train' / 'images',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_test_datagen.flow_from_directory(
            self.data_path / 'val' / 'images',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.data_path / 'test' / 'images',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Train samples: {self.train_generator.samples}")
        print(f"✅ Validation samples: {self.val_generator.samples}")
        print(f"✅ Test samples: {self.test_generator.samples}")
        
        # Calculate class weights for imbalanced data
        labels = self.train_generator.labels
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weight_dict = dict(enumerate(class_weights))
        print(f"📊 Class weights: {self.class_weight_dict}")
    
    def build_model(self, architecture='efficientnet'):
        """Build CNN model with transfer learning"""
        print(f"\n🏗️ Building {architecture} model...")
        
        # Input layer
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Choose base model
        if architecture == 'efficientnet':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                pooling='avg'
            )
            # Unfreeze last 20 layers
            for layer in base_model.layers[:-20]:
                layer.trainable = False
        else:  # ResNet
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                pooling='avg'
            )
            # Unfreeze last 30 layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
        
        # Get base model output
        x = base_model.output
        
        # Add custom layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        print(f"✅ Model built with {len(self.model.layers)} layers")
        print(f"   Trainable parameters: {self.model.count_params():,}")
    
    def train_model(self, epochs=20):
        """Train the model"""
        print(f"\n🚀 Training model for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                self.model_path / 'best_image_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            class_weight=self.class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training complete!")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📊 Evaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.test_generator)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = self.test_generator.labels
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multi-class, calculate metrics with averaging
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Store results
        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_class_accuracy': {}
        }
        
        # Per-class accuracy
        cm = confusion_matrix(y_true, y_pred)
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
                self.results['per_class_accuracy'][f'grade_{i}'] = float(class_acc)
        
        print(f"\n📊 Test Results:")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"\n  Per-class Accuracy:")
        for grade, acc in self.results['per_class_accuracy'].items():
            print(f"    {grade}: {acc:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Grade {i}' for i in range(5)],
                   yticklabels=[f'Grade {i}' for i in range(5)])
        plt.title('Confusion Matrix - Image Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('docs/image_model_confusion_matrix.png')
        plt.show()
        
    def plot_training_history(self):
        """Plot training history"""
        print("\n📈 Plotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('docs/image_model_training_history.png')
        plt.show()
    
    def save_model_artifacts(self):
        """Save model and metadata"""
        print("\n💾 Saving model artifacts...")
        
        # Save model
        self.model.save(self.model_path / 'image_model_final.h5')
        
        # Save results
        with open(self.model_path / 'image_model_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'architecture': 'EfficientNetB0',
            'input_size': list(self.img_size),
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs_trained': len(self.history.history['loss']),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'test_accuracy': self.results['accuracy']
        }
        
        with open(self.model_path / 'image_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ All artifacts saved!")

def main():
    print("👁️ Image Model Training Pipeline")
    print("="*50)
    
    trainer = ImageModelTrainer()
    
    # Pipeline
    trainer.prepare_data_generators()
    trainer.build_model('efficientnet')
    trainer.train_model(epochs=10)  # Reduced epochs for faster training
    trainer.evaluate_model()
    trainer.plot_training_history()
    trainer.save_model_artifacts()
    
    print("\n" + "="*50)
    print("✨ Image model training complete!")
    print(f"📊 Test Accuracy: {trainer.results['accuracy']:.4f}")
    print("\nNext step: Run 'python ml-pipeline/src/train_fusion.py'")

if __name__ == "__main__":
    main()