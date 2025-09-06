"""
Production testing suite for your DR detection system
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import time

class ProductionModelTester:
    def __init__(self):
        self.model_path = Path("ml-pipeline/models")
        
    def test_inference_speed(self, model_path):
        """Test if model meets latency requirements"""
        model = tf.keras.models.load_model(model_path)
        
        # Test batch
        test_batch = np.random.random((1, 224, 224, 3))
        
        # Warmup
        _ = model.predict(test_batch, verbose=0)
        
        # Time inference
        times = []
        for _ in range(100):
            start = time.time()
            _ = model.predict(test_batch, verbose=0)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        
        assert avg_time < 200, f"Inference too slow: {avg_time:.2f}ms"
        print(f"✅ Inference speed: {avg_time:.2f}ms (< 200ms required)")
        
        return avg_time
    
    def test_model_consistency(self, model_path):
        """Test if model gives consistent predictions"""
        model = tf.keras.models.load_model(model_path)
        
        # Same image, multiple predictions
        test_image = np.random.random((1, 224, 224, 3))
        
        predictions = []
        for _ in range(10):
            pred = model.predict(test_image, verbose=0)
            predictions.append(pred[0])
        
        # Check variance
        variance = np.var(predictions)
        assert variance < 0.001, f"Predictions inconsistent: variance={variance}"
        print(f"✅ Model consistency: variance={variance:.6f}")
        
    def test_edge_cases(self, model_path):
        """Test model with edge cases"""
        model = tf.keras.models.load_model(model_path)
        
        test_cases = {
            "black_image": np.zeros((1, 224, 224, 3)),
            "white_image": np.ones((1, 224, 224, 3)),
            "noisy_image": np.random.random((1, 224, 224, 3))
        }
        
        for case_name, test_image in test_cases.items():
            pred = model.predict(test_image, verbose=0)
            assert 0 <= pred[0][0] <= 1, f"Invalid prediction for {case_name}"
            print(f"✅ {case_name}: {pred[0][0]:.4f}")

# Run tests
tester = ProductionModelTester()
tester.test_inference_speed("ml-pipeline/models/stable_dr_ensemble.keras")
tester.test_model_consistency("ml-pipeline/models/stable_dr_ensemble.keras")
tester.test_edge_cases("ml-pipeline/models/stable_dr_ensemble.keras")