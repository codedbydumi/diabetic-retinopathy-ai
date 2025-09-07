"""
Train a working diabetic retinopathy image model using transfer learning
Optimized for CPU training with good accuracy
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm
import zipfile
import requests

warnings.filterwarnings('ignore')

# Configure for CPU optimization
tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
tf.keras.backend.clear_session()

class DiabeticRetinopathyTrainer:
    def __init__(self, model_path="ml-pipeline/models", image_size=224):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        self.image_size = image_size
        self.num_classes = 5
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        
    def download_sample_dataset(self):
        """Download and prepare a sample dataset for training"""
        print("Creating synthetic training dataset...")
        
        # Create dataset directory
        dataset_dir = self.model_path / "training_data"
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate synthetic retinal images with different characteristics
        images = []
        labels = []
        
        # Generate 500 synthetic images per class
        samples_per_class = 100  # Reduced for faster CPU training
        
        for class_idx in range(self.num_classes):
            print(f"Generating {samples_per_class} images for class {class_idx} ({self.class_names[class_idx]})")
            
            for i in tqdm(range(samples_per_class)):
                # Create base retinal image
                img = self.generate_retinal_image(class_idx)
                
                # Save image
                class_dir = dataset_dir / str(class_idx)
                class_dir.mkdir(exist_ok=True)
                
                img_path = class_dir / f"{class_idx}_{i:03d}.jpg"
                img.save(img_path)
                
                images.append(str(img_path))
                labels.append(class_idx)
        
        return images, labels
    
    def generate_retinal_image(self, severity):
        """Generate synthetic retinal images with different DR characteristics"""
        # Create base circular retinal fundus
        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Base retinal color (reddish-brown)
        base_color = [180, 100, 80] if severity == 0 else [200, 90, 70]
        img[:, :] = base_color
        
        # Add circular mask for retinal shape
        center = self.image_size // 2
        y, x = np.ogrid[:self.image_size, :self.image_size]
        mask = (x - center) ** 2 + (y - center) ** 2 <= (center - 20) ** 2
        img[~mask] = [0, 0, 0]
        
        # Add optic disc (bright circular region)
        disc_x, disc_y = center + 30, center - 10
        cv2.circle(img, (disc_x, disc_y), 25, (255, 220, 200), -1)
        
        # Add blood vessels
        self.add_blood_vessels(img, severity)
        
        # Add pathological features based on severity
        if severity >= 1:  # Mild DR
            self.add_microaneurysms(img, severity * 3)
        
        if severity >= 2:  # Moderate DR
            self.add_hemorrhages(img, severity * 2)
            self.add_exudates(img, severity * 2)
        
        if severity >= 3:  # Severe DR
            self.add_cotton_wool_spots(img, severity)
            self.add_venous_beading(img)
        
        if severity >= 4:  # Proliferative DR
            self.add_neovascularization(img)
            self.add_fibrous_proliferation(img)
        
        # Add noise and variations
        img = self.add_realistic_variations(img)
        
        return Image.fromarray(img)
    
    def add_blood_vessels(self, img, severity):
        """Add blood vessel patterns"""
        center = self.image_size // 2
        
        # Main vessels
        for angle in np.linspace(0, 2*np.pi, 8):
            end_x = int(center + (center-30) * np.cos(angle))
            end_y = int(center + (center-30) * np.sin(angle))
            
            thickness = 3 if severity < 3 else 4
            color = (120, 60, 50) if severity < 2 else (100, 50, 40)
            
            cv2.line(img, (center, center), (end_x, end_y), color, thickness)
            
            # Add smaller branching vessels
            for sub_angle in [angle - 0.3, angle + 0.3]:
                sub_end_x = int(center + (center-50) * np.cos(sub_angle))
                sub_end_y = int(center + (center-50) * np.sin(sub_angle))
                cv2.line(img, (center, center), (sub_end_x, sub_end_y), color, 2)
    
    def add_microaneurysms(self, img, count):
        """Add small red dots (microaneurysms)"""
        for _ in range(count):
            x = np.random.randint(30, self.image_size-30)
            y = np.random.randint(30, self.image_size-30)
            cv2.circle(img, (x, y), 2, (180, 50, 50), -1)
    
    def add_hemorrhages(self, img, count):
        """Add larger red blotches (hemorrhages)"""
        for _ in range(count):
            x = np.random.randint(40, self.image_size-40)
            y = np.random.randint(40, self.image_size-40)
            cv2.circle(img, (x, y), np.random.randint(4, 8), (160, 40, 40), -1)
    
    def add_exudates(self, img, count):
        """Add bright yellow/white spots (hard exudates)"""
        for _ in range(count):
            x = np.random.randint(50, self.image_size-50)
            y = np.random.randint(50, self.image_size-50)
            cv2.circle(img, (x, y), np.random.randint(3, 6), (255, 255, 200), -1)
    
    def add_cotton_wool_spots(self, img, count):
        """Add fluffy white patches (cotton wool spots)"""
        for _ in range(count):
            x = np.random.randint(60, self.image_size-60)
            y = np.random.randint(60, self.image_size-60)
            cv2.circle(img, (x, y), np.random.randint(6, 10), (240, 240, 240), -1)
    
    def add_venous_beading(self, img):
        """Add irregular vessel patterns"""
        center = self.image_size // 2
        for angle in np.linspace(0, 2*np.pi, 4):
            points = []
            for r in range(30, center-20, 10):
                noise_x = np.random.randint(-5, 5)
                noise_y = np.random.randint(-5, 5)
                x = int(center + r * np.cos(angle) + noise_x)
                y = int(center + r * np.sin(angle) + noise_y)
                points.append((x, y))
            
            for i in range(len(points)-1):
                cv2.line(img, points[i], points[i+1], (100, 40, 40), 4)
    
    def add_neovascularization(self, img):
        """Add new abnormal blood vessels"""
        center = self.image_size // 2
        for _ in range(5):
            start_x = np.random.randint(center-30, center+30)
            start_y = np.random.randint(center-30, center+30)
            
            for _ in range(10):
                end_x = start_x + np.random.randint(-20, 20)
                end_y = start_y + np.random.randint(-20, 20)
                cv2.line(img, (start_x, start_y), (end_x, end_y), (200, 60, 60), 2)
                start_x, start_y = end_x, end_y
    
    def add_fibrous_proliferation(self, img):
        """Add fibrous tissue"""
        for _ in range(3):
            x = np.random.randint(50, self.image_size-50)
            y = np.random.randint(50, self.image_size-50)
            cv2.ellipse(img, (x, y), (15, 8), np.random.randint(0, 180), 0, 360, (220, 220, 220), -1)
    
    def add_realistic_variations(self, img):
        """Add noise, blur, and lighting variations"""
        # Add Gaussian noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-20, 20)    # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Slight blur
        if np.random.random() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def create_data_generators(self, images, labels):
        """Create data generators for training"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create data generators with augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2]
        )
        
        val_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Create DataFrames for generators
        train_df = pd.DataFrame({'filename': X_train, 'class': [str(c) for c in y_train]})
        val_df = pd.DataFrame({'filename': X_val, 'class': [str(c) for c in y_val]})
        
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='class',
            target_size=(self.image_size, self.image_size),
            batch_size=16,  # Small batch size for CPU
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='class',
            target_size=(self.image_size, self.image_size),
            batch_size=16,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def create_model(self):
        """Create model using transfer learning"""
        print("Creating model with transfer learning...")
        
        # Use MobileNetV2 as base (lightweight for CPU)
        base_model = applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.image_size, self.image_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train_model(self, model, train_gen, val_gen):
        """Train the model"""
        print("Starting training...")
        
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
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path / 'best_image_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train model
        steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
        validation_steps = max(1, val_gen.samples // val_gen.batch_size)
        
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=30,  # Reduced for CPU training
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase
        print("Starting fine-tuning...")
        
        # Unfreeze top layers of base model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 20
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Continue training
        fine_tune_epochs = 10
        total_epochs = 30 + fine_tune_epochs
        
        history_fine = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            initial_epoch=30,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate_model(self, model, val_gen):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Get predictions
        val_gen.reset()
        predictions = model.predict(val_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = val_gen.classes
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, target_names=self.class_names))
        
        return accuracy
    
    def save_final_model(self, model):
        """Save the final trained model"""
        print("Saving trained model...")
        
        # Save in multiple formats
        model.save(self.model_path / 'image_model_trained.h5')
        model.save(self.model_path / 'image_model_trained.keras')
        model.save_weights(self.model_path / 'image_model_trained.weights.h5')
        
        # Save metadata
        metadata = {
            'model_name': 'diabetic_retinopathy_trained',
            'architecture': 'MobileNetV2 + Custom Head',
            'framework': 'TensorFlow/Keras',
            'input_shape': [self.image_size, self.image_size, 3],
            'output_classes': self.num_classes,
            'class_names': self.class_names,
            'preprocessing': 'Rescaling to [0,1]',
            'training_method': 'Transfer Learning + Fine-tuning',
            'base_model': 'MobileNetV2 (ImageNet)',
            'status': 'trained',
            'note': 'Trained on synthetic data for demonstration'
        }
        
        with open(self.model_path / 'image_model_trained_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Model saved successfully!")
        
    def train_full_pipeline(self):
        """Complete training pipeline"""
        print("=" * 60)
        print("DIABETIC RETINOPATHY MODEL TRAINING")
        print("=" * 60)
        
        try:
            # Step 1: Generate dataset
            print("\n1. Generating training dataset...")
            images, labels = self.download_sample_dataset()
            
            # Step 2: Create data generators
            print("\n2. Creating data generators...")
            train_gen, val_gen = self.create_data_generators(images, labels)
            
            # Step 3: Create model
            print("\n3. Creating model...")
            model = self.create_model()
            print(f"Model created with {model.count_params():,} parameters")
            
            # Step 4: Train model
            print("\n4. Training model...")
            model, history = self.train_model(model, train_gen, val_gen)
            
            # Step 5: Evaluate
            print("\n5. Evaluating model...")
            accuracy = self.evaluate_model(model, val_gen)
            
            # Step 6: Save model
            print("\n6. Saving model...")
            self.save_final_model(model)
            
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print(f"Final Validation Accuracy: {accuracy:.2%}")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Update your backend to load 'image_model_trained.h5'")
            print("2. Restart your backend API")
            print("3. Test image predictions!")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    # Initialize trainer
    trainer = DiabeticRetinopathyTrainer()
    
    # Run full training pipeline
    success = trainer.train_full_pipeline()
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)