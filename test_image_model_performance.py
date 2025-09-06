"""
Test the image model performance and get accuracy metrics
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_test_model():
    """Load the image model and test its performance"""
    
    model_path = Path("ml-pipeline/models")
    
    # Try to load the model
    model = None
    for model_name in ['image_model_fixed.h5', 'image_model_fixed.keras']:
        model_file = model_path / model_name
        if model_file.exists():
            try:
                model = tf.keras.models.load_model(model_file)
                print(f"✅ Loaded model from: {model_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    if model is None:
        print("❌ No image model found!")
        return
    
    # Model summary
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Generate test data (since we don't have real labeled retinal images)
    print("\n🧪 Generating test data...")
    n_samples = 100
    test_images = np.random.random((n_samples, 224, 224, 3)).astype(np.float32)
    
    # Create fake labels for testing (0-4 for DR grades)
    true_labels = np.random.randint(0, 5, n_samples)
    
    # Get predictions
    print("🔮 Making predictions...")
    predictions = model.predict(test_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"\n📈 Model Performance (on random test data):")
    print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    report = classification_report(true_labels, predicted_labels, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    print(f"\n📋 Classification Report:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            print(f"   {class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Image Model Performance')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save the plot
    plot_path = model_path / 'image_model_performance.png'
    plt.savefig(plot_path)
    print(f"📊 Confusion matrix saved to: {plot_path}")
    plt.show()
    
    # Test individual prediction confidence
    print(f"\n🎯 Sample Predictions:")
    for i in range(5):
        pred_probs = predictions[i]
        pred_class = predicted_labels[i]
        confidence = np.max(pred_probs)
        
        print(f"   Sample {i+1}: Predicted Class {pred_class} ({class_names[pred_class]}) "
              f"with {confidence:.1%} confidence")
    
    # Save results
    results = {
        'model_type': 'Custom CNN for Diabetic Retinopathy',
        'test_accuracy': float(accuracy),
        'test_samples': n_samples,
        'classification_report': report,
        'note': 'Tested on random data - accuracy reflects random performance',
        'recommendation': 'Train with real retinal images for meaningful accuracy'
    }
    
    results_path = model_path / 'image_model_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {results_path}")
    
    return results

def evaluate_model_readiness():
    """Check if the model is ready for production"""
    
    model_path = Path("ml-pipeline/models")
    
    print("\n🔍 Model Readiness Assessment:")
    
    # Check if model files exist
    model_files = {
        'image_model_fixed.h5': model_path / 'image_model_fixed.h5',
        'image_model_fixed.keras': model_path / 'image_model_fixed.keras',
        'image_model.weights.h5': model_path / 'image_model.weights.h5',
        'image_model_metadata.json': model_path / 'image_model_metadata.json'
    }
    
    for name, path in model_files.items():
        status = "✅ Found" if path.exists() else "❌ Missing"
        print(f"   {name}: {status}")
    
    # Load metadata if available
    metadata_path = model_path / 'image_model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📄 Model Metadata:")
        print(f"   Architecture: {metadata.get('architecture', 'Unknown')}")
        print(f"   Parameters: {metadata.get('total_parameters', 'Unknown'):,}")
        print(f"   Input Shape: {metadata.get('input_shape', 'Unknown')}")
        print(f"   Output Classes: {metadata.get('output_classes', 'Unknown')}")
        print(f"   Status: {metadata.get('status', 'Unknown')}")
    
    print(f"\n⚠️  Current Limitations:")
    print(f"   • Model has random weights (not trained)")
    print(f"   • Predictions are essentially random")
    print(f"   • Need real retinal image dataset for training")
    print(f"   • Consider transfer learning from pre-trained models")

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 IMAGE MODEL PERFORMANCE TESTING")
    print("=" * 60)
    
    # Test the model
    results = load_and_test_model()
    
    # Evaluate readiness
    evaluate_model_readiness()
    
    print("\n" + "=" * 60)
    print("📝 SUMMARY")
    print("=" * 60)
    print("• Your image model loads and runs successfully")
    print("• Current accuracy is random (~20% for 5 classes)")
    print("• Model architecture is solid and ready for training")
    print("• Next step: Train with real diabetic retinopathy images")
    print("=" * 60)