"""
Fix the model file naming and compatibility issues
"""

import shutil
from pathlib import Path
import tensorflow as tf
import json

def fix_model():
    model_path = Path("ml-pipeline/models")
    
    print("🔧 Fixing model files...")
    
    # Check what files exist
    if (model_path / "best_image_model.h5").exists():
        print("   Found best_image_model.h5")
        
        # Copy to correct name
        shutil.copy(
            model_path / "best_image_model.h5",
            model_path / "retina_image_model.h5"
        )
        print("   ✅ Created retina_image_model.h5")
    else:
        print("   ⚠️ best_image_model.h5 not found")
    
    # Create a simple mock model if needed (for testing fusion)
    if not (model_path / "retina_image_model.h5").exists():
        print("   Creating mock image model for testing...")
        
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.save(model_path / "retina_image_model.h5")
        print("   ✅ Created mock retina_image_model.h5")
    
    print("✅ Model files fixed!")

if __name__ == "__main__":
    fix_model()