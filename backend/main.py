"""
FastAPI Backend for Diabetic Retinopathy Detection System
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
from PIL import Image
import io
import json
from datetime import datetime
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Multi-modal AI system for diabetic retinopathy detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
MODELS = {}
MODEL_PATH = Path("ml-pipeline/models")

# ==================== Pydantic Models ====================

class ClinicalData(BaseModel):
    """Clinical data input model"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Glucose level")
    blood_pressure: float = Field(..., ge=0, le=200, description="Blood pressure")
    skin_thickness: float = Field(..., ge=0, le=100, description="Skin thickness")
    insulin: float = Field(..., ge=0, le=900, description="Insulin level")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    diabetes_pedigree: float = Field(..., ge=0, le=3, description="Diabetes pedigree function")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    hba1c: float = Field(..., ge=4, le=15, description="HbA1c level")
    cholesterol: float = Field(..., ge=100, le=400, description="Cholesterol level")
    smoking: bool = Field(..., description="Smoking status")
    family_history: bool = Field(..., description="Family history of diabetes")
    exercise_weekly: int = Field(..., ge=0, le=7, description="Days of exercise per week")
    
    class Config:
        schema_extra = {
            "example": {
                "pregnancies": 2,
                "glucose": 120.0,
                "blood_pressure": 70.0,
                "skin_thickness": 20.0,
                "insulin": 80.0,
                "bmi": 25.5,
                "diabetes_pedigree": 0.5,
                "age": 35,
                "hba1c": 5.5,
                "cholesterol": 200.0,
                "smoking": False,
                "family_history": True,
                "exercise_weekly": 3
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction_id: str
    timestamp: str
    clinical_risk: float
    image_risk: Optional[float] = None
    combined_risk: float
    risk_level: str
    confidence: float
    recommendations: List[str]
    feature_importance: Optional[Dict[str, float]] = None
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    available_models: List[str]
    timestamp: str

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

# ==================== Model Loading ====================

def load_models():
    """Load all trained models"""
    global MODELS
    
    try:
        logger.info("Loading models...")
        
        # Load clinical model
        MODELS['clinical_scaler'] = joblib.load(MODEL_PATH / 'clinical_scaler.pkl')
        MODELS['clinical_model'] = joblib.load(MODEL_PATH / 'clinical_ensemble.pkl')
        logger.info("✅ Clinical model loaded")
        
        # Load fusion model
        MODELS['fusion_model'] = joblib.load(MODEL_PATH / 'fusion_model.pkl')
        logger.info("✅ Fusion model loaded")
        
        # Try to load image model
        try:
            if (MODEL_PATH / 'image_model_final.h5').exists():
                MODELS['image_model'] = tf.keras.models.load_model(
                    MODEL_PATH / 'image_model_final.h5',
                    compile=False
                )
                MODELS['image_model'].compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Image model loaded")
            else:
                logger.warning("⚠️ Image model not found")
                MODELS['image_model'] = None
        except Exception as e:
            logger.error(f"Error loading image model: {e}")
            MODELS['image_model'] = None
        
        # Load model metadata
        with open(MODEL_PATH / 'clinical_metadata.json', 'r') as f:
            MODELS['clinical_metadata'] = json.load(f)
        
        with open(MODEL_PATH / 'fusion_metadata.json', 'r') as f:
            MODELS['fusion_metadata'] = json.load(f)
        
        logger.info("✅ All models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

# ==================== Utility Functions ====================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def get_risk_level(risk_score: float) -> str:
    """Convert risk score to risk level"""
    if risk_score < 0.3:
        return "Low"
    elif risk_score < 0.6:
        return "Medium"
    elif risk_score < 0.8:
        return "High"
    else:
        return "Very High"

def get_recommendations(risk_level: str, clinical_data: ClinicalData) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    # Base recommendations by risk level
    if risk_level in ["High", "Very High"]:
        recommendations.append("🚨 Schedule an appointment with an ophthalmologist immediately")
        recommendations.append("📊 Get comprehensive diabetic eye exam within 2 weeks")
    elif risk_level == "Medium":
        recommendations.append("📅 Schedule eye screening within 3 months")
        recommendations.append("👁️ Monitor for vision changes")
    else:
        recommendations.append("✅ Continue annual eye screenings")
    
    # Personalized recommendations based on clinical data
    if clinical_data.glucose > 140:
        recommendations.append("🩸 Work on better glucose control (current: {:.1f})".format(clinical_data.glucose))
    
    if clinical_data.bmi > 30:
        recommendations.append("⚖️ Consider weight management program (BMI: {:.1f})".format(clinical_data.bmi))
    
    if clinical_data.smoking:
        recommendations.append("🚭 Smoking cessation strongly recommended")
    
    if clinical_data.exercise_weekly < 3:
        recommendations.append("🏃 Increase physical activity to at least 3 days/week")
    
    if clinical_data.hba1c > 7:
        recommendations.append("📈 HbA1c is elevated ({:.1f}%) - discuss with your doctor".format(clinical_data.hba1c))
    
    return recommendations

def calculate_feature_importance(clinical_data: ClinicalData) -> Dict[str, float]:
    """Calculate feature importance for the prediction"""
    # Get feature importances from the model
    if hasattr(MODELS.get('clinical_model'), 'feature_importances_'):
        feature_names = MODELS['clinical_metadata']['features']
        importances = MODELS['clinical_model'].feature_importances_
        
        # Create importance dictionary
        importance_dict = {}
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:5])
        return importance_dict
    
    return {}

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    success = load_models()
    if not success:
        logger.error("Failed to load models on startup")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetic Retinopathy Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        'clinical_model' in MODELS,
        'fusion_model' in MODELS
    ])
    
    available_models = [
        model for model in ['clinical_model', 'image_model', 'fusion_model'] 
        if model in MODELS and MODELS[model] is not None
    ]
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        available_models=available_models,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict/clinical", response_model=PredictionResponse)
async def predict_clinical(data: ClinicalData):
    """Predict using clinical data only"""
    try:
        # Prepare features
        features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree, data.age, data.hba1c,
            data.cholesterol, int(data.smoking), int(data.family_history),
            data.exercise_weekly
        ]])
        
        # Scale features
        features_scaled = MODELS['clinical_scaler'].transform(features)
        
        # Get prediction
        risk_prob = MODELS['clinical_model'].predict_proba(features_scaled)[0, 1]
        
        # Determine risk level
        risk_level = get_risk_level(risk_prob)
        
        # Get recommendations
        recommendations = get_recommendations(risk_level, data)
        
        # Get feature importance
        feature_importance = calculate_feature_importance(data)
        
        return PredictionResponse(
            prediction_id=f"PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            clinical_risk=float(risk_prob),
            combined_risk=float(risk_prob),
            risk_level=risk_level,
            confidence=float(0.85),  # Based on model accuracy
            recommendations=recommendations,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/combined", response_model=PredictionResponse)
async def predict_combined(
    clinical_data: ClinicalData = Depends(),
    image: Optional[UploadFile] = File(None)
):
    """Predict using both clinical and image data"""
    try:
        # Get clinical prediction
        clinical_features = np.array([[
            clinical_data.pregnancies, clinical_data.glucose, clinical_data.blood_pressure,
            clinical_data.skin_thickness, clinical_data.insulin, clinical_data.bmi,
            clinical_data.diabetes_pedigree, clinical_data.age, clinical_data.hba1c,
            clinical_data.cholesterol, int(clinical_data.smoking), 
            int(clinical_data.family_history), clinical_data.exercise_weekly
        ]])
        
        clinical_scaled = MODELS['clinical_scaler'].transform(clinical_features)
        clinical_prob = MODELS['clinical_model'].predict_proba(clinical_scaled)[0, 1]
        
        # Process image if provided
        image_prob = None
        if image and MODELS.get('image_model'):
            image_bytes = await image.read()
            img_array = preprocess_image(image_bytes)
            
            # Get image prediction
            img_pred = MODELS['image_model'].predict(img_array)
            # Convert multi-class to binary (grades 2+ indicate risk)
            image_prob = float(np.sum(img_pred[0, 2:]))  # Sum of moderate to proliferative
        
        # Combine predictions
        if image_prob is not None:
            # Use fusion model
            fusion_features = np.array([[
                clinical_prob,
                image_prob,
                clinical_prob * image_prob,
                np.abs(clinical_prob - image_prob),
                max(clinical_prob, image_prob),
                min(clinical_prob, image_prob),
                (clinical_prob + image_prob) / 2,
                clinical_prob ** 2,
                image_prob ** 2
            ]])
            
            combined_risk = MODELS['fusion_model'].predict_proba(fusion_features)[0, 1]
            confidence = 0.92  # Based on fusion model accuracy
        else:
            # Use clinical only
            combined_risk = clinical_prob
            confidence = 0.85
        
        risk_level = get_risk_level(combined_risk)
        recommendations = get_recommendations(risk_level, clinical_data)
        
        return PredictionResponse(
            prediction_id=f"PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            clinical_risk=float(clinical_prob),
            image_risk=float(image_prob) if image_prob else None,
            combined_risk=float(combined_risk),
            risk_level=risk_level,
            confidence=float(confidence),
            recommendations=recommendations,
            feature_importance=calculate_feature_importance(clinical_data)
        )
        
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/metrics", response_model=List[ModelMetrics])
async def get_model_metrics():
    """Get performance metrics for all models"""
    metrics = []
    
    # Load saved results
    with open(MODEL_PATH / 'clinical_results.json', 'r') as f:
        clinical_results = json.load(f)
    
    with open(MODEL_PATH / 'fusion_results.json', 'r') as f:
        fusion_results = json.load(f)
    
    # Clinical model metrics
    metrics.append(ModelMetrics(
        model_type="Clinical (Random Forest)",
        accuracy=clinical_results['random_forest']['accuracy'],
        precision=clinical_results['random_forest']['precision'],
        recall=clinical_results['random_forest']['recall'],
        f1_score=clinical_results['random_forest']['f1'],
        auc_roc=clinical_results['random_forest']['auc_roc']
    ))
    
    # Fusion model metrics
    best_fusion = max(fusion_results.items(), key=lambda x: x[1]['accuracy'])
    metrics.append(ModelMetrics(
        model_type=f"Fusion ({best_fusion[0].replace('_', ' ').title()})",
        accuracy=best_fusion[1]['accuracy'],
        precision=best_fusion[1]['precision'],
        recall=best_fusion[1]['recall'],
        f1_score=best_fusion[1]['f1'],
        auc_roc=best_fusion[1]['auc_roc']
    ))
    
    return metrics

@app.get("/models/info")
async def get_models_info():
    """Get information about loaded models"""
    info = {
        "clinical_model": {
            "loaded": 'clinical_model' in MODELS,
            "type": "Ensemble (XGBoost + Random Forest)",
            "features": MODELS.get('clinical_metadata', {}).get('features', []),
            "accuracy": 0.878
        },
        "image_model": {
            "loaded": MODELS.get('image_model') is not None,
            "type": "MobileNetV2",
            "input_size": [224, 224, 3],
            "classes": 5
        },
        "fusion_model": {
            "loaded": 'fusion_model' in MODELS,
            "type": MODELS.get('fusion_metadata', {}).get('best_strategy', 'Unknown'),
            "accuracy": MODELS.get('fusion_metadata', {}).get('best_accuracy', 0)
        }
    }
    return info

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )