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
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
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
        self.img_size = (224, 224)  # Standard input size
        self.batch_size = 32
        self.num_classes = 5  # DR grades 0-4
        
        self.model = None
        self.history = None
        self.results = {}
        
    def create_synthetic_images(self):
        """
        Create synthetic RGB images for training
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
                
                # Create a synthetic RGB image
                img = self.create_synthetic_retina_image(grade)
                
                # Save image
                img_path = grade_dir / f"{row['image_id']}.png"
                tf.keras.preprocessing.image.save_img(img_path, img)
            
            print(f"   ✅ Created {len(df)} images for {split} set")
    
    def create_synthetic_retina_image(self, grade):
        """
        Create a synthetic retinal RGB image based on DR grade
        """
        # Create base circular pattern resembling retina (RGB)
        img = np.zeros((*self.img_size, 3), dtype=np.float32)
        center = (self.img_size[0] // 2, self.img_size[1] // 2)
        
        # Create radial gradient for retina appearance
        y, x = np.ogrid[:self.img_size[0], :self.img_size[1]]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        max_dist = np.sqrt(center[0]**2 + center[1]**2)
        
        # Base retina appearance (reddish circular pattern)
        mask = dist_from_center <= 100
        intensity = np.where(mask, 1.0 - (dist_from_center / 100), 0)
        
        # Create RGB channels with different patterns per grade
        # Red channel - base retina
        img[:, :, 0] = intensity * (0.7 - grade * 0.1)
        # Green channel - slightly less
        img[:, :, 1] = intensity * (0.3 - grade * 0.05)
        # Blue channel - least
        img[:, :, 2] = intensity * (0.2 - grade * 0.03)
        
        # Add synthetic blood vessels
        np.random.seed(42 + grade)
        num_vessels = 5 + grade
        for _ in range(num_vessels):
            # Create curved lines to simulate vessels
            t = np.linspace(0, 2*np.pi, 100)
            cx = center[0] + 50 * np.cos(t) * np.random.uniform(0.5, 1.5)
            cy = center[1] + 50 * np.sin(t) * np.random.uniform(0.5, 1.5)
            
            for i in range(len(t)-1):
                x1, y1 = int(cx[i]), int(cy[i])
                if 0 <= x1 < self.img_size[0] and 0 <= y1 < self.img_size[1]:
                    img[y1, x1] = [0.4, 0.1, 0.1]
        
        # Add grade-specific features
        if grade > 0:
            # Add microaneurysms (small red dots)
            num_spots = grade * 5
            for _ in range(num_spots):
                x = np.random.randint(20, self.img_size[0]-20)
                y = np.random.randint(20, self.img_size[1]-20)
                if mask[y, x]:
                    cv_size = 2 if grade < 3 else 3
                    img[max(0, y-cv_size):min(self.img_size[0], y+cv_size),
                        max(0, x-cv_size):min(self.img_size[1], x+cv_size)] = [0.8, 0.2, 0.1]
        
        if grade > 2:
            # Add hemorrhages (larger dark spots)
            num_hemorrhages = (grade - 2) * 3
            for _ in range(num_hemorrhages):
                x = np.random.randint(30, self.img_size[0]-30)
                y = np.random.randint(30, self.img_size[1]-30)
                if mask[y, x]:
                    img[max(0, y-5):min(self.img_size[0], y+5),
                        max(0, x-5):min(self.img_size[1], x+5)] = [0.5, 0.1, 0.05]
        
        # Add noise for realism
        noise = np.random.normal(0, 0.02, img.shape)
        img = img + noise
        
        # Ensure values are in [0, 1] range for RGB
        img = np.clip(img, 0, 1)
        
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
            vertical_flip=False,  # Don't flip vertically for medical images
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
    
    def build_model(self, architecture='mobilenet'):
        """Build CNN model with transfer learning"""
        print(f"\n🏗️ Building {architecture} model...")
        
        # Clear any existing models
        tf.keras.backend.clear_session()
        
        # Build model using Functional API
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layer (part of model)
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        x = data_augmentation(inputs)
        
        # Preprocessing for the specific model
        if architecture == 'mobilenet':
            # Use MobileNetV2 which is more stable
            preprocess_input = keras.applications.mobilenet_v2.preprocess_input
            x = preprocess_input(x)
            
            base_model = keras.applications.MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            # Freeze base model layers initially
            base_model.trainable = False
            
        else:  # Use ResNet50V2 as alternative
            preprocess_input = keras.applications.resnet_v2.preprocess_input
            x = preprocess_input(x)
            
            base_model = keras.applications.ResNet50V2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            base_model.trainable = False
        
        # Pass through base model
        x = base_model(x, training=False)
        
        # Add custom classification head
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer for 5 classes
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create the model
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print(f"✅ Model built successfully!")
        print(f"   Total layers: {len(self.model.layers)}")
        print(f"   Trainable parameters: {sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights):,}")
        print(f"   Non-trainable parameters: {sum(tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights):,}")
        
        # Store base model reference for fine-tuning
        self.base_model = base_model
    
    def train_model(self, epochs=15):
        """Train the model with two-stage training"""
        print(f"\n🚀 Starting two-stage training...")
        
        # Stage 1: Train only the top layers
        print("\n📌 Stage 1: Training classification head only...")
        
        callbacks_stage1 = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        history_stage1 = self.model.fit(
            self.train_generator,
            epochs=5,  # Quick initial training
            validation_data=self.val_generator,
            class_weight=self.class_weight_dict,
            callbacks=callbacks_stage1,
            verbose=1
        )
        
        # Stage 2: Fine-tune the model
        print("\n📌 Stage 2: Fine-tuning with unfrozen base layers...")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(self.base_model.layers) - 20
        
        # Freeze all layers before fine_tune_at
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        callbacks_stage2 = [
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
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        history_stage2 = self.model.fit(
            self.train_generator,
            epochs=epochs - 5,  # Remaining epochs
            validation_data=self.val_generator,
            class_weight=self.class_weight_dict,
            callbacks=callbacks_stage2,
            verbose=1
        )
        
        # Combine histories
        self.history = history_stage2
        
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
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
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
            if i < len(cm) and cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum()
                self.results['per_class_accuracy'][f'grade_{i}'] = float(class_acc)
            else:
                self.results['per_class_accuracy'][f'grade_{i}'] = 0.0
        
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
            'architecture': 'MobileNetV2',
            'input_size': list(self.img_size),
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'epochs_trained': len(self.history.history['loss']) if self.history else 0,
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]) if self.history else 0,
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
    trainer.build_model('mobilenet')  # Using MobileNet for stability
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