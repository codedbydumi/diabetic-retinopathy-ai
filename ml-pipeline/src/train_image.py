"""
Image Model Training Script
Train CNN model for retinal image classification using transfer learning
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImageModelTrainer:
    def __init__(self):
        self.data_path = Path("ml-pipeline/data")
        self.model_path = Path("ml-pipeline/models")
        self.model_path.mkdir(exist_ok=True)
        
        # Image parameters
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.num_classes = 5  # DR grades 0-4
        
        self.model = None
        self.history = None
        self.results = {}
        
    def create_synthetic_images(self):
        """
        Create synthetic images for training
        In production, you would load real retinal images
        """
        print("🖼️ Creating synthetic training images...")
        
        # Load metadata
        train_meta = pd.read_csv(self.data_path / "train" / "image_metadata_train.csv")
        val_meta = pd.read_csv(self.data_path / "val" / "image_metadata_val.csv")
        test_meta = pd.read_csv(self.data_path / "test" / "image_metadata_test.csv")
        
        # Create synthetic images for each grade
        for split, metadata in [('train', train_meta), ('val', val_meta), ('test', test_meta)]:
            split_path = self.data_path / split
            
            for _, row in metadata.iterrows():
                grade = row['dr_grade']
                grade_dir = split_path / f'grade_{grade}'
                grade_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a synthetic image (in production, load real image)
                # Different patterns for different grades
                img = self.generate_synthetic_retina_image(grade)
                
                # Save image
                img_path = grade_dir / f"{row['image_id']}.jpg"
                tf.keras.preprocessing.image.save_img(img_path, img)
            
            print(f"   ✅ Created {len(metadata)} images for {split} set")
    
    def generate_synthetic_retina_image(self, grade):
        """
        Generate a synthetic retina-like image based on DR grade
        This is for demonstration - real images would be loaded instead
        """
        # Create base circular pattern
        img = np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        center = (self.img_height // 2, self.img_width // 2)
        
        # Create circular mask (retina shape)
        Y, X = np.ogrid[:self.img_height, :self.img_width]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist_from_center <= 100
        
        # Base retina color (reddish)
        img[mask] = [0.8, 0.3, 0.2]
        
        # Add features based on grade
        noise_level = 0.1 + (grade * 0.05)
        
        # Add noise to simulate pathology
        noise = np.random.normal(0, noise_level, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        # Add some blood vessel-like patterns
        for _ in range(5 + grade * 2):
            start_x = np.random.randint(50, 150)
            start_y = np.random.randint(50, 150)
            end_x = np.random.randint(50, 150)
            end_y = np.random.randint(50, 150)
            
            # Simple line drawing (in production, use actual vessel patterns)
            rr, cc = np.linspace(start_y, end_y, 50).astype(int), \
                     np.linspace(start_x, end_x, 50).astype(int)
            
            valid_indices = (rr >= 0) & (rr < self.img_height) & \
                          (cc >= 0) & (cc < self.img_width)
            rr, cc = rr[valid_indices], cc[valid_indices]
            
            if len(rr) > 0:
                img[rr, cc] = [0.4, 0.1, 0.1]
        
        # Add spots for higher grades (simulate microaneurysms/hemorrhages)
        if grade >= 2:
            num_spots = grade * 3
            for _ in range(num_spots):
                spot_x = np.random.randint(40, 160)
                spot_y = np.random.randint(40, 160)
                radius = np.random.randint(2, 5)
                
                Y, X = np.ogrid[:self.img_height, :self.img_width]
                dist = np.sqrt((X - spot_x)**2 + (Y - spot_y)**2)
                spot_mask = dist <= radius
                
                img[spot_mask] = [0.9, 0.8, 0.2] if grade >= 3 else [0.7, 0.2, 0.2]
        
        return img
    
    def prepare_data_generators(self):
        """Prepare data generators for training"""
        print("\n📊 Preparing data generators...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='constant',
            cval=0
        )
        
        # Only rescaling for validation/test
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path / 'train',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path / 'val',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_datagen.flow_from_directory(
            self.data_path / 'test',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"   ✅ Train samples: {self.train_generator.samples}")
        print(f"   ✅ Validation samples: {self.val_generator.samples}")
        print(f"   ✅ Test samples: {self.test_generator.samples}")
    
    def build_model(self, model_type='efficientnet'):
        """Build CNN model with transfer learning"""
        print(f"\n🏗️ Building {model_type} model...")
        
        # Input layer
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Choose base model
        if model_type == 'efficientnet':
            base_model = EfficientNetB3(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                pooling='avg'
            )
            base_model.trainable = False  # Freeze base model initially
        else:  # resnet
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs,
                pooling='avg'
            )
            base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        print("   ✅ Model built successfully")
        print(f"   Total parameters: {self.model.count_params():,}")
        
        # Save model architecture
        model_json = self.model.to_json()
        with open(self.model_path / f'{model_type}_architecture.json', 'w') as f:
            f.write(model_json)
    
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
            callbacks=callbacks,
            verbose=1
        )
        
        print("   ✅ Training completed")
    
    def fine_tune_model(self, epochs=10):
        """Fine-tune the model by unfreezing some layers"""
        print("\n🎯 Fine-tuning model...")
        
        # Unfreeze the top layers of the base model
        for layer in self.model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Continue training
        history_fine = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(history_fine.history[key])
        
        print("   ✅ Fine-tuning completed")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📊 Evaluating model...")
        
        # Test set evaluation
        test_loss, test_acc, test_auc = self.model.evaluate(
            self.test_generator,
            verbose=0
        )
        
        print(f"   Test Accuracy: {test_acc:.4f}")
        print(f"   Test AUC: {test_auc:.4f}")
        print(f"   Test Loss: {test_loss:.4f}")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.test_generator)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = self.test_generator.classes
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Image Model - Confusion Matrix')
        plt.xlabel('Predicted Grade')
        plt.ylabel('True Grade')
        plt.savefig('docs/image_model_confusion_matrix.png')
        plt.show()
        
        # Classification report
        class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Save results
        self.results = {
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_loss': float(test_loss),
            'classification_report': report
        }
        
        with open(self.model_path / 'image_model_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
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
    
    def save_model(self):
        """Save the trained model"""
        print("\n💾 Saving model...")
        
        # Save model
        self.model.save(self.model_path / 'retina_image_model.h5')
        
        # Save as TensorFlow SavedModel format (for production)
        tf.saved_model.save(self.model, str(self.model_path / 'image_model_saved'))
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'image_size': [self.img_height, self.img_width],
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'total_epochs': len(self.history.history['loss']),
            'final_accuracy': float(self.history.history['val_accuracy'][-1]),
            'test_accuracy': self.results['test_accuracy']
        }
        
        with open(self.model_path / 'image_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model saved to {self.model_path}")

def main():
    print("👁️ Retinal Image Model Training Pipeline")
    print("="*50)
    
    trainer = ImageModelTrainer()
    
    # Create synthetic images (in production, load real images)
    trainer.create_synthetic_images()
    
    # Prepare data
    trainer.prepare_data_generators()
    
    # Build and train model
    trainer.build_model('efficientnet')
    trainer.train_model(epochs=10)  # Reduced epochs for faster training
    trainer.fine_tune_model(epochs=5)
    
    # Evaluate
    trainer.evaluate_model()
    trainer.plot_training_history()
    trainer.save_model()
    
    print("\n" + "="*50)
    print("✨ Image model training complete!")
    print(f"   Final Test Accuracy: {trainer.results['test_accuracy']:.4f}")
    print("\nNext step: Run 'python ml-pipeline/src/train_fusion.py'")

if __name__ == "__main__":
    main()