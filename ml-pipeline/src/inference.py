"""
Inference Pipeline
Unified interface for making predictions with the multi-modal system
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
from PIL import Image
import json
import time
from typing import Dict, Optional, Tuple

class DiabetesRiskPredictor:
    def __init__(self):
        self.model_path = Path("ml-pipeline/models")
        self.models_loaded = False
        
        # Models
        self.clinical_model = None
        self.image_model = None
        self.fusion_model = None
        self.scaler = None
        
        # Configuration
        self.fusion_config = None
        self.pipeline_config = None
        
    def load_models(self):
        """Load all models and configurations"""
        print("📦 Loading models...")
        start_time = time.time()
        
        try:
            # Load clinical model
            self.clinical_model = joblib.load(self.model_path / 'clinical_ensemble.pkl')
            print("   ✅ Clinical model loaded")
            
            # Load image model
            self.image_model = tf.keras.models.load_model(
                self.model_path / 'retina_image_model.h5',
                compile=False
            )
            self.image_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("   ✅ Image model loaded")
            
            # Load scaler
            self.scaler = joblib.load(self.model_path / 'clinical_scaler.pkl')
            print("   ✅ Scaler loaded")
            
            # Load fusion configuration
            fusion_model_path = self.model_path / 'fusion_model.h5'
            if fusion_model_path.exists():
                self.fusion_model = tf.keras.models.load_model(fusion_model_path)
                print("   ✅ Neural fusion model loaded")
            else:
                with open(self.model_path / 'fusion_config.json', 'r') as f:
                    self.fusion_config = json.load(f)
                print("   ✅ Fusion configuration loaded")
            
            # Load pipeline configuration
            with open(self.model_path / 'pipeline_config.json', 'r') as f:
                self.pipeline_config = json.load(f)
            
            self.models_loaded = True
            load_time = time.time() - start_time
            print(f"   ⏱️ Models loaded in {load_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise
    
    def preprocess_clinical_data(self, clinical_data: Dict) -> np.ndarray:
        """Preprocess clinical data for prediction"""
        # Expected features
        expected_features = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'hba1c',
            'cholesterol', 'smoking', 'family_history', 'exercise_weekly'
        ]
        
        # Create dataframe
        df = pd.DataFrame([clinical_data])
        
        # Ensure all features are present
        for feature in expected_features:
            if feature not in df.columns:
                # Set default values for missing features
                if feature in ['smoking', 'family_history']:
                    df[feature] = 0
                else:
                    df[feature] = df.mean().mean()  # Use mean of other features
        
        # Select and order features
        df = df[expected_features]
        
        # Scale features
        scaled_data = self.scaler.transform(df)
        
        return scaled_data
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess retinal image for prediction"""
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_clinical(self, clinical_data: Dict) -> Tuple[float, Dict]:
        """Make prediction using clinical data only"""
        # Preprocess
        processed_data = self.preprocess_clinical_data(clinical_data)
        
        # Predict
        prob = self.clinical_model.predict_proba(processed_data)[0, 1]
        prediction = int(prob > 0.5)
        
        # Get feature importance (simplified)
        feature_names = list(clinical_data.keys())
        
        return prob, {
            'probability': float(prob),
            'prediction': prediction,
            'risk_level': self._get_risk_level(prob),
            'confidence': self._calculate_confidence(prob),
            'model_type': 'clinical'
        }
    
    def predict_image(self, image_path: str) -> Tuple[float, Dict]:
        """Make prediction using retinal image only"""
        # Preprocess
        img_array = self.preprocess_image(image_path)
        
        # Predict DR grade
        dr_probs = self.image_model.predict(img_array)[0]
        dr_grade = np.argmax(dr_probs)
        
        # Convert DR grade to diabetes risk
        # Higher DR grades indicate higher diabetes risk
        grade_to_risk = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 0.95}
        diabetes_prob = grade_to_risk[dr_grade]
        
        # Add some uncertainty based on confidence
        confidence = float(np.max(dr_probs))
        diabetes_prob = diabetes_prob * confidence + 0.5 * (1 - confidence)
        
        return diabetes_prob, {
            'probability': float(diabetes_prob),
            'dr_grade': int(dr_grade),
            'dr_probabilities': dr_probs.tolist(),
            'prediction': int(diabetes_prob > 0.5),
            'risk_level': self._get_risk_level(diabetes_prob),
            'confidence': confidence,
            'model_type': 'image'
        }
    
    def predict_fusion(self, clinical_data: Dict, image_path: Optional[str] = None) -> Dict:
        """Make prediction using fusion of clinical and image data"""
        if not self.models_loaded:
            self.load_models()
        
        start_time = time.time()
        
        # Get clinical prediction
        clinical_prob, clinical_results = self.predict_clinical(clinical_data)
        
        # Get image prediction if available
        if image_path and Path(image_path).exists():
            image_prob, image_results = self.predict_image(image_path)
            
            # Fusion prediction
            if self.fusion_model:
                # Neural fusion
                fusion_input = np.array([[clinical_prob, image_prob]])
                fusion_prob = float(self.fusion_model.predict(fusion_input)[0, 0])
            else:
                # Simple/weighted fusion
                if self.fusion_config['method'] == 'Weighted Average':
                    fusion_prob = (
                        self.fusion_config['clinical_weight'] * clinical_prob +
                        self.fusion_config['image_weight'] * image_prob
                    )
                elif self.fusion_config['method'] == 'Maximum':
                    fusion_prob = max(clinical_prob, image_prob)
                else:  # Simple Average
                    fusion_prob = (clinical_prob + image_prob) / 2
            
            # Combine results
            results = {
                'fusion_probability': float(fusion_prob),
                'fusion_prediction': int(fusion_prob > 0.5),
                'risk_level': self._get_risk_level(fusion_prob),
                'confidence': self._calculate_confidence(fusion_prob),
                'clinical': clinical_results,
                'image': image_results,
                'fusion_method': self.fusion_config.get('method', 'Neural Fusion'),
                'processing_time': time.time() - start_time
            }
        else:
            # Clinical only
            results = {
                'fusion_probability': float(clinical_prob),
                'fusion_prediction': int(clinical_prob > 0.5),
                'risk_level': self._get_risk_level(clinical_prob),
                'confidence': self._calculate_confidence(clinical_prob),
                'clinical': clinical_results,
                'image': None,
                'fusion_method': 'Clinical Only',
                'processing_time': time.time() - start_time
            }
        
        # Add recommendations
        results['recommendations'] = self._get_recommendations(results['risk_level'])
        
        return results
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.6:
            return "Moderate"
        elif probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence score"""
        # Higher confidence when probability is far from 0.5
        distance_from_center = abs(probability - 0.5) * 2
        return min(0.95, 0.6 + distance_from_center * 0.35)
    
    def _get_recommendations(self, risk_level: str) -> list:
        """Get recommendations based on risk level"""
        recommendations = {
            "Low": [
                "Maintain healthy lifestyle",
                "Regular annual check-ups",
                "Continue balanced diet and exercise"
            ],
            "Moderate": [
                "Schedule comprehensive diabetes screening",
                "Monitor blood glucose levels regularly",
                "Increase physical activity to 150 minutes/week",
                "Consider dietary consultation"
            ],
            "High": [
                "Immediate medical consultation recommended",
                "Comprehensive metabolic panel needed",
                "Regular blood glucose monitoring",
                "Lifestyle intervention program",
                "Possible medication evaluation"
            ],
            "Very High": [
                "Urgent medical attention required",
                "Immediate comprehensive diabetes evaluation",
                "Start glucose monitoring program",
                "Medication likely necessary",
                "Referral to endocrinologist recommended"
            ]
        }
        
        return recommendations.get(risk_level, [])

def main():
    """Test the inference pipeline"""
    print("🔬 Testing Inference Pipeline")
    print("="*50)
    
    # Initialize predictor
    predictor = DiabetesRiskPredictor()
    predictor.load_models()
    
    # Test data
    test_clinical_data = {
        'pregnancies': 2,
        'glucose': 120,
        'blood_pressure': 70,
        'skin_thickness': 30,
        'insulin': 100,
        'bmi': 28.5,
        'diabetes_pedigree': 0.5,
        'age': 35,
        'hba1c': 5.8,
        'cholesterol': 200,
        'smoking': 0,
        'family_history': 1,
        'exercise_weekly': 3
    }
    
    # Make prediction
    print("\n📊 Making prediction...")
    results = predictor.predict_fusion(
        clinical_data=test_clinical_data,
        image_path=None  # No image for this test
    )
    
    # Display results
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(f"Risk Probability: {results['fusion_probability']:.2%}")
    print(f"Risk Level: {results['risk_level']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"\nRecommendations:")
    for rec in results['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ Inference pipeline test complete!")

if __name__ == "__main__":
    main()