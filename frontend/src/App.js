/**
 * Diabetic Retinopathy Detection System - React Frontend
 * Complete single-file React application
 */

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
axios.defaults.baseURL = API_BASE_URL;

// ==================== MAIN APP COMPONENT ====================
const App = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    checkHealth();
    getModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get('/health');
      setHealthStatus(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const getModelInfo = async () => {
    try {
      const response = await axios.get('/models/info');
      setModelInfo(response.data);
    } catch (error) {
      console.error('Failed to get model info:', error);
    }
  };

  return (
    <div className="App">
      <Header healthStatus={healthStatus} />
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        {activeTab === 'predict' && <PredictionForm />}
        {activeTab === 'about' && <AboutSection modelInfo={modelInfo} />}
        {activeTab === 'metrics' && <MetricsSection />}
      </main>
      <Footer />
    </div>
  );
};

// ==================== HEADER COMPONENT ====================
const Header = ({ healthStatus }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="logo-section">
          <span className="logo">👁️</span>
          <h1>Diabetic Retinopathy AI Detection</h1>
        </div>
        <div className="status-badge">
          <span className={`badge ${healthStatus?.status === 'healthy' ? 'badge-success' : 'badge-warning'}`}>
            {healthStatus?.status === 'healthy' ? '🟢 System Online' : '🟡 Limited Mode'}
          </span>
        </div>
      </div>
    </header>
  );
};

// ==================== NAVIGATION COMPONENT ====================
const Navigation = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="navigation">
      <button 
        className={`nav-button ${activeTab === 'predict' ? 'active' : ''}`}
        onClick={() => setActiveTab('predict')}
      >
        🔬 Predict
      </button>
      <button 
        className={`nav-button ${activeTab === 'metrics' ? 'active' : ''}`}
        onClick={() => setActiveTab('metrics')}
      >
        📊 Model Metrics
      </button>
      <button 
        className={`nav-button ${activeTab === 'about' ? 'active' : ''}`}
        onClick={() => setActiveTab('about')}
      >
        ℹ️ About
      </button>
    </nav>
  );
};

// ==================== PREDICTION FORM COMPONENT ====================
const PredictionForm = () => {
  const [formData, setFormData] = useState({
    pregnancies: 2,
    glucose: 120,
    blood_pressure: 70,
    skin_thickness: 20,
    insulin: 80,
    bmi: 25.5,
    diabetes_pedigree: 0.5,
    age: 35,
    hba1c: 5.5,
    cholesterol: 200,
    smoking: false,
    family_history: false,
    exercise_weekly: 3
  });

  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : parseFloat(value) || value
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      let response;
      
      if (imageFile) {
        const formDataToSend = new FormData();
        formDataToSend.append('data', JSON.stringify(formData));
        formDataToSend.append('image', imageFile);
        
        response = await axios.post('/predict/combined', formDataToSend, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
      } else {
        response = await axios.post('/predict/clinical', formData);
      }

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      'Low': '#2ecc71',
      'Medium': '#f39c12',
      'High': '#e74c3c',
      'Very High': '#c0392b'
    };
    return colors[level] || '#95a5a6';
  };

  return (
    <div className="prediction-section">
      <div className="form-container">
        <h2>Patient Information</h2>
        <form onSubmit={handleSubmit} className="prediction-form">
          <div className="form-grid">
            {/* Demographics */}
            <div className="form-section">
              <h3>Demographics</h3>
              <div className="input-group">
                <label>Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  min="18"
                  max="100"
                  required
                />
              </div>
              <div className="input-group">
                <label>Pregnancies</label>
                <input
                  type="number"
                  name="pregnancies"
                  value={formData.pregnancies}
                  onChange={handleInputChange}
                  min="0"
                  max="20"
                />
              </div>
            </div>

            {/* Clinical Measurements */}
            <div className="form-section">
              <h3>Clinical Measurements</h3>
              <div className="input-group">
                <label>Glucose (mg/dL)</label>
                <input
                  type="number"
                  name="glucose"
                  value={formData.glucose}
                  onChange={handleInputChange}
                  min="0"
                  max="300"
                  step="0.1"
                  required
                />
              </div>
              <div className="input-group">
                <label>Blood Pressure (mm Hg)</label>
                <input
                  type="number"
                  name="blood_pressure"
                  value={formData.blood_pressure}
                  onChange={handleInputChange}
                  min="0"
                  max="200"
                  step="0.1"
                />
              </div>
              <div className="input-group">
                <label>BMI</label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleInputChange}
                  min="10"
                  max="70"
                  step="0.1"
                  required
                />
              </div>
              <div className="input-group">
                <label>HbA1c (%)</label>
                <input
                  type="number"
                  name="hba1c"
                  value={formData.hba1c}
                  onChange={handleInputChange}
                  min="4"
                  max="15"
                  step="0.1"
                  required
                />
              </div>
            </div>

            {/* Lab Values */}
            <div className="form-section">
              <h3>Lab Values</h3>
              <div className="input-group">
                <label>Insulin (μU/mL)</label>
                <input
                  type="number"
                  name="insulin"
                  value={formData.insulin}
                  onChange={handleInputChange}
                  min="0"
                  max="900"
                  step="0.1"
                />
              </div>
              <div className="input-group">
                <label>Cholesterol (mg/dL)</label>
                <input
                  type="number"
                  name="cholesterol"
                  value={formData.cholesterol}
                  onChange={handleInputChange}
                  min="100"
                  max="400"
                  step="0.1"
                />
              </div>
              <div className="input-group">
                <label>Skin Thickness (mm)</label>
                <input
                  type="number"
                  name="skin_thickness"
                  value={formData.skin_thickness}
                  onChange={handleInputChange}
                  min="0"
                  max="100"
                  step="0.1"
                />
              </div>
              <div className="input-group">
                <label>Diabetes Pedigree</label>
                <input
                  type="number"
                  name="diabetes_pedigree"
                  value={formData.diabetes_pedigree}
                  onChange={handleInputChange}
                  min="0"
                  max="3"
                  step="0.001"
                />
              </div>
            </div>

            {/* Lifestyle */}
            <div className="form-section">
              <h3>Lifestyle Factors</h3>
              <div className="input-group">
                <label>Exercise (days/week)</label>
                <input
                  type="number"
                  name="exercise_weekly"
                  value={formData.exercise_weekly}
                  onChange={handleInputChange}
                  min="0"
                  max="7"
                />
              </div>
              <div className="checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="smoking"
                    checked={formData.smoking}
                    onChange={handleInputChange}
                  />
                  <span>Smoking</span>
                </label>
              </div>
              <div className="checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="family_history"
                    checked={formData.family_history}
                    onChange={handleInputChange}
                  />
                  <span>Family History of Diabetes</span>
                </label>
              </div>
            </div>
          </div>

          {/* Image Upload */}
          <div className="image-upload-section">
            <h3>Retinal Image (Optional)</h3>
            <div className="upload-container">
              <input
                type="file"
                id="image-upload"
                accept="image/*"
                onChange={handleImageChange}
                className="file-input"
              />
              <label htmlFor="image-upload" className="upload-label">
                📷 Choose Retinal Image
              </label>
              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="Preview" />
                  <button 
                    type="button" 
                    onClick={() => {
                      setImageFile(null);
                      setImagePreview(null);
                    }}
                    className="remove-image"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Submit Button */}
          <button 
            type="submit" 
            className="submit-button"
            disabled={loading}
          >
            {loading ? '🔄 Analyzing...' : '🔬 Analyze Risk'}
          </button>
        </form>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* Prediction Results */}
        {prediction && (
          <div className="results-container">
            <h2>Analysis Results</h2>
            
            <div className="risk-summary">
              <div 
                className="risk-level-card"
                style={{ borderColor: getRiskColor(prediction.risk_level) }}
              >
                <h3>Risk Level</h3>
                <div 
                  className="risk-value"
                  style={{ color: getRiskColor(prediction.risk_level) }}
                >
                  {prediction.risk_level}
                </div>
                <div className="confidence">
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </div>
              </div>

              <div className="risk-scores">
                <div className="score-item">
                  <span className="score-label">Clinical Risk:</span>
                  <span className="score-value">
                    {(prediction.clinical_risk * 100).toFixed(1)}%
                  </span>
                </div>
                {prediction.image_risk !== null && (
                  <div className="score-item">
                    <span className="score-label">Image Risk:</span>
                    <span className="score-value">
                      {(prediction.image_risk * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                <div className="score-item">
                  <span className="score-label">Combined Risk:</span>
                  <span className="score-value">
                    {(prediction.combined_risk * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="recommendations">
              <h3>📋 Recommendations</h3>
              <ul>
                {prediction.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>

            {prediction.feature_importance && (
              <div className="feature-importance">
                <h3>🔍 Key Risk Factors</h3>
                <div className="importance-bars">
                  {Object.entries(prediction.feature_importance)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 5)
                    .map(([feature, importance]) => (
                      <div key={feature} className="importance-item">
                        <span className="feature-name">
                          {feature.replace(/_/g, ' ').toUpperCase()}
                        </span>
                        <div className="importance-bar-container">
                          <div 
                            className="importance-bar"
                            style={{ width: `${importance * 100}%` }}
                          />
                        </div>
                        <span className="importance-value">
                          {(importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            <div className="prediction-id">
              Prediction ID: {prediction.prediction_id}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ==================== METRICS SECTION COMPONENT ====================
const MetricsSection = () => {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('/models/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading metrics...</div>;
  }

  return (
    <div className="metrics-section">
      <h2>Model Performance Metrics</h2>
      <div className="metrics-grid">
        {metrics.map((model, idx) => (
          <div key={idx} className="metric-card">
            <h3>{model.model_type}</h3>
            <div className="metric-values">
              <div className="metric-item">
                <span className="metric-label">Accuracy</span>
                <span className="metric-value">
                  {(model.accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Precision</span>
                <span className="metric-value">
                  {(model.precision * 100).toFixed(2)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Recall</span>
                <span className="metric-value">
                  {(model.recall * 100).toFixed(2)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">F1-Score</span>
                <span className="metric-value">
                  {(model.f1_score * 100).toFixed(2)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">AUC-ROC</span>
                <span className="metric-value">
                  {model.auc_roc.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// ==================== ABOUT SECTION COMPONENT ====================
const AboutSection = ({ modelInfo }) => {
  return (
    <div className="about-section">
      <h2>About This System</h2>
      
      <div className="about-content">
        <div className="about-card">
          <h3>🎯 Purpose</h3>
          <p>
            This AI-powered system combines clinical data and retinal imaging to detect 
            diabetic retinopathy risk. It uses multi-modal deep learning to provide 
            accurate risk assessments and personalized recommendations.
          </p>
        </div>

        <div className="about-card">
          <h3>🤖 Technology Stack</h3>
          <ul>
            <li><strong>Backend:</strong> FastAPI, Python 3.9+</li>
            <li><strong>ML Models:</strong> XGBoost, Random Forest, ResNet50, VGG16, MobileNetV2</li>
            <li><strong>Image Processing:</strong> TensorFlow, OpenCV, CLAHE preprocessing</li>
            <li><strong>Frontend:</strong> React, Axios</li>
            <li><strong>Deployment:</strong> Docker, Kubernetes-ready</li>
          </ul>
        </div>

        <div className="about-card">
          <h3>📊 Model Information</h3>
          {modelInfo && (
            <div className="model-info-grid">
              <div className="model-info-item">
                <h4>Clinical Model</h4>
                <p>Status: {modelInfo.clinical_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</p>
                <p>Type: {modelInfo.clinical_model?.type}</p>
                <p>Accuracy: {(modelInfo.clinical_model?.accuracy * 100).toFixed(1)}%</p>
              </div>
              <div className="model-info-item">
                <h4>Image Model</h4>
                <p>Status: {modelInfo.image_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</p>
                <p>Type: {modelInfo.image_model?.type}</p>
                <p>Current: {modelInfo.image_model?.current_model || 'Unknown'}</p>
                <p>Components: {modelInfo.image_model?.ensemble_components?.join(', ') || 'N/A'}</p>
              </div>
              <div className="model-info-item">
                <h4>Fusion Model</h4>
                <p>Status: {modelInfo.fusion_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</p>
                <p>Type: {modelInfo.fusion_model?.type}</p>
                <p>Accuracy: {modelInfo.fusion_model?.accuracy ? (modelInfo.fusion_model.accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
              </div>
            </div>
          )}
        </div>

        <div className="about-card">
          <h3>⚠️ Disclaimer</h3>
          <p className="disclaimer">
            This system is for educational and research purposes only. It should not be used 
            as a substitute for professional medical advice, diagnosis, or treatment. Always 
            consult with qualified healthcare providers for medical decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

// ==================== FOOTER COMPONENT ====================
const Footer = () => {
  return (
    <footer className="app-footer">
      <div className="footer-content">
        <p>© 2025 Diabetic Retinopathy AI Detection System</p>
        <p>Built with ❤️ by Dumindu Thushan for improving healthcare accessibility through AI</p>
      </div>
    </footer>
  );
};

export default App;