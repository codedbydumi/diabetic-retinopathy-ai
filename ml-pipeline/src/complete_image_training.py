"""
Complete the image model training (evaluation and saving)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def complete_training():
    print("📊 Completing image model training...")
    
    # Paths
    data_path = Path("ml-pipeline/data")
    model_path = Path("ml-pipeline/models")
    
    # Load the saved model
    model = tf.keras.models.load_model(model_path / 'best_image_model.h5')
    print("✅ Model loaded successfully")
    
    # Prepare test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        data_path / 'test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    test_loss, test_acc, test_auc = model.evaluate(test_generator, verbose=0)
    
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test AUC: {test_auc:.4f}")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Get predictions
    y_pred_proba = model.predict(test_generator)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_generator.classes
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    Path("docs").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Image Model - Confusion Matrix')
    plt.xlabel('Predicted Grade')
    plt.ylabel('True Grade')
    plt.savefig('docs/image_model_confusion_matrix.png')
    plt.close()
    
    # Classification report
    class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save final model with correct name
    model.save(model_path / 'retina_image_model.h5')
    print(f"\n✅ Model saved as retina_image_model.h5")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss),
        'classification_report': report
    }
    
    with open(model_path / 'image_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'image_size': [224, 224],
        'num_classes': 5,
        'batch_size': 32,
        'test_accuracy': float(test_acc),
        'model_type': 'MobileNetV2'
    }
    
    with open(model_path / 'image_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✨ Image model training completed successfully!")
    print(f"   Final Test Accuracy: {test_acc:.4f}")
    print("\nNext step: Run 'python ml-pipeline/src/train_fusion.py'")

if __name__ == "__main__":
    complete_training()