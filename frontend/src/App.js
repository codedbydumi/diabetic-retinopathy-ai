import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { ToastContainer, useToast } from './components/Toast';
import ErrorBoundary from './components/ErrorBoundary';
import { PredictionLoading } from './components/LoadingStates';


// API Configuration
// In App.js, temporarily use a CORS proxy
//const API_BASE_URL = 'http://5.189.151.50:8003';
//axios.defaults.baseURL = API_BASE_URL;
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api'  // Vercel will proxy this
  : 'http://localhost:8000';

axios.defaults.baseURL = API_BASE_URL;

// Performance Chart Component
const PerformanceChart = ({ performanceHistory }) => {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  if (!performanceHistory || performanceHistory.length === 0) {
    return (
      <div className="chart-placeholder">
        <p>No performance data available</p>
      </div>
    );
  }

  const maxValue = Math.max(
    ...performanceHistory.flatMap(d => [d.clinical, d.image, d.fusion])
  );
  const minValue = Math.min(
    ...performanceHistory.flatMap(d => [d.clinical, d.image, d.fusion])
  );
  
  const chartHeight = 200;
  const chartWidth = 500;
  const padding = { top: 20, right: 20, bottom: 40, left: 60 };
  
  const getY = (value) => {
    const range = maxValue - minValue || 0.1;
    return chartHeight - padding.bottom - ((value - minValue) / range) * (chartHeight - padding.top - padding.bottom);
  };
  
  const getX = (index) => {
    return padding.left + (index / (performanceHistory.length - 1)) * (chartWidth - padding.left - padding.right);
  };

  const createPath = (key) => {
    return performanceHistory
      .map((d, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)} ${getY(d[key])}`)
      .join(' ');
  };

  const colors = {
    clinical: '#4CAF50',
    image: '#2196F3', 
    fusion: '#FF9800'
  };

  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  return (
    <div style={{ background: 'white', borderRadius: '8px', padding: '1rem', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <h4 style={{ margin: '0 0 1rem 0', color: '#333', textAlign: 'center' }}>Model Accuracy Over Time</h4>
      
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
        <svg width={chartWidth} height={chartHeight} style={{ border: '1px solid #e0e0e0', borderRadius: '4px' }}>
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="20" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Y-axis labels */}
          {[0.8, 0.85, 0.9, 0.95].map(value => (
            <g key={value}>
              <text 
                x={padding.left - 10} 
                y={getY(value) + 4} 
                textAnchor="end" 
                fontSize="12" 
                fill="#666"
              >
                {(value * 100).toFixed(0)}%
              </text>
              <line 
                x1={padding.left} 
                y1={getY(value)} 
                x2={chartWidth - padding.right} 
                y2={getY(value)} 
                stroke="#e0e0e0" 
                strokeWidth="1"
              />
            </g>
          ))}
          
          {/* X-axis labels */}
          {performanceHistory.map((d, i) => (
            <text 
              key={i}
              x={getX(i)} 
              y={chartHeight - 10} 
              textAnchor="middle" 
              fontSize="10" 
              fill="#666"
            >
              {formatDate(d.date)}
            </text>
          ))}
          
          {/* Lines */}
          {['clinical', 'image', 'fusion'].map(key => (
            <path
              key={key}
              d={createPath(key)}
              fill="none"
              stroke={colors[key]}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          
          {/* Data points */}
          {performanceHistory.map((d, i) => 
            ['clinical', 'image', 'fusion'].map(key => (
              <circle
                key={`${i}-${key}`}
                cx={getX(i)}
                cy={getY(d[key])}
                r="4"
                fill={colors[key]}
                stroke="white"
                strokeWidth="2"
                style={{ cursor: 'pointer', transition: 'r 0.2s ease' }}
                onMouseEnter={() => setHoveredPoint({ index: i, key, data: d })}
                onMouseLeave={() => setHoveredPoint(null)}
              />
            ))
          )}
          
          {/* Tooltip */}
          {hoveredPoint && (
            <g>
              <rect
                x={getX(hoveredPoint.index) + 10}
                y={getY(hoveredPoint.data[hoveredPoint.key]) - 30}
                width="120"
                height="50"
                fill="rgba(0,0,0,0.8)"
                rx="4"
              />
              <text
                x={getX(hoveredPoint.index) + 15}
                y={getY(hoveredPoint.data[hoveredPoint.key]) - 15}
                fill="white"
                fontSize="10"
              >
                {hoveredPoint.data.date}
              </text>
              <text
                x={getX(hoveredPoint.index) + 15}
                y={getY(hoveredPoint.data[hoveredPoint.key]) - 5}
                fill="white"
                fontSize="10"
                fontWeight="bold"
              >
                {hoveredPoint.key}: {(hoveredPoint.data[hoveredPoint.key] * 100).toFixed(1)}%
              </text>
            </g>
          )}
        </svg>
      </div>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        {Object.entries(colors).map(([key, color]) => (
          <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', color: '#666' }}>
            <span 
              style={{ 
                width: '12px', 
                height: '12px', 
                backgroundColor: color, 
                borderRadius: '2px' 
              }}
            ></span>
            <span style={{ fontWeight: '500' }}>
              {key.charAt(0).toUpperCase() + key.slice(1)} Model
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [healthStatus, setHealthStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const { toasts, addToast, removeToast } = useToast();

  useEffect(() => {
    checkHealth();
    getModelInfo();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await axios.get('/health');
      setHealthStatus(response.data);
      if (response.data.status === 'degraded') {
        addToast('warning', 'System running in limited mode');
      }
    } catch (error) {
      console.error('Health check failed:', error);
      addToast('error', 'Failed to connect to backend');
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
    <ErrorBoundary>
      <div className="App">
        <Header healthStatus={healthStatus} />
        <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="main-content">
          {activeTab === 'metrics' && <MetricsSection />}
          {activeTab === 'predict' && <PredictionForm addToast={addToast} />}
          
          {activeTab === 'about' && <AboutSection modelInfo={modelInfo} />}
        </main>
        <Footer />
        <ToastContainer toasts={toasts} removeToast={removeToast} />
      </div>
    </ErrorBoundary>
  );
};

// Header Component
const Header = ({ healthStatus }) => {
  return (
    <header className="app-header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo">👁️</div>
          <div>
            <h1>DR Detection System</h1>
            <p className="tagline">AI-Powered Diabetic Retinopathy Analysis</p>
          </div>
        </div>
        <div className="header-stats">
          <div className="stat-item">
            <span className="stat-label">Status</span>
            <span className={`status-badge ${healthStatus?.status === 'healthy' ? 'status-healthy' : 'status-degraded'}`}>
              {healthStatus?.status === 'healthy' ? '● Online' : '● Limited'}
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Models</span>
            <span className="stat-value">{healthStatus?.available_models?.length || 0}/3</span>
          </div>
        </div>
      </div>
    </header>
  );
};

// Navigation Component
const Navigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'metrics', label: 'Metrics', icon: '📊' },
    { id: 'predict', label: 'Analyze', icon: '🔬' },
    
    { id: 'about', label: 'About', icon: 'ℹ️' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-container">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`nav-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="nav-icon">{tab.icon}</span>
            <span className="nav-label">{tab.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
};

// Prediction Form Component
const PredictionForm = ({ addToast }) => {
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
  const [errors, setErrors] = useState({});

  const validateField = (name, value) => {
    const validationRules = {
      glucose: { min: 0, max: 500, message: 'Glucose must be between 0-500 mg/dL' },
      age: { min: 18, max: 120, message: 'Age must be between 18-120 years' },
      bmi: { min: 10, max: 70, message: 'BMI must be between 10-70' },
      blood_pressure: { min: 0, max: 250, message: 'Blood pressure must be between 0-250 mmHg' },
      hba1c: { min: 4, max: 15, message: 'HbA1c must be between 4-15%' },
      insulin: { min: 0, max: 900, message: 'Insulin must be between 0-900 μU/mL' },
      cholesterol: { min: 100, max: 400, message: 'Cholesterol must be between 100-400 mg/dL' },
      skin_thickness: { min: 0, max: 100, message: 'Skin thickness must be between 0-100 mm' },
      diabetes_pedigree: { min: 0, max: 3, message: 'Diabetes pedigree must be between 0-3' },
      exercise_weekly: { min: 0, max: 7, message: 'Exercise days must be between 0-7' }
    };

    if (validationRules[name]) {
      const rule = validationRules[name];
      if (value < rule.min || value > rule.max) {
        return rule.message;
      }
    }
    return null;
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : parseFloat(value) || value;
    
    setFormData(prev => ({
      ...prev,
      [name]: newValue
    }));

    // Validate on change
    if (type !== 'checkbox') {
      const error = validateField(name, newValue);
      setErrors(prev => ({
        ...prev,
        [name]: error
      }));
    }
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.size > 10 * 1024 * 1024) {
        addToast('error', 'Image size must be less than 10MB');
        return;
      }
      
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
      addToast('success', 'Image uploaded successfully');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate all fields
    const newErrors = {};
    Object.keys(formData).forEach(key => {
      const error = validateField(key, formData[key]);
      if (error) newErrors[key] = error;
    });

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      addToast('error', 'Please correct the errors in the form');
      return;
    }

    setLoading(true);
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
      addToast('success', 'Analysis completed successfully');
    } catch (err) {
      console.error('Prediction error:', err);
      addToast('error', err.response?.data?.detail || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level) => {
    const colors = {
      'Low': '#00A651',
      'Medium': '#FFB800',
      'High': '#DC3545',
      'Very High': '#8B0000'
    };
    return colors[level] || '#6C757D';
  };

  if (loading) {
    return <PredictionLoading />;
  }

  return (
    <div className="prediction-section">
      <div className="section-header">
        <h2>Patient Risk Assessment</h2>
        <p>Enter clinical data and optionally upload a retinal image for analysis</p>
      </div>

      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-grid">
          {/* Demographics Section */}
          <div className="form-section">
            <h3>Demographics</h3>
            <div className="input-row">
              <div className="input-group">
                <label htmlFor="age">
                  Age <span className="required">*</span>
                </label>
                <input
                  id="age"
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleInputChange}
                  className={errors.age ? 'error' : ''}
                  required
                />
                {errors.age && <span className="error-message">{errors.age}</span>}
              </div>
              
              <div className="input-group">
                <label htmlFor="pregnancies">Pregnancies</label>
                <input
                  id="pregnancies"
                  type="number"
                  name="pregnancies"
                  value={formData.pregnancies}
                  onChange={handleInputChange}
                  min="0"
                  max="20"
                />
              </div>
            </div>
          </div>

          {/* Clinical Measurements Section */}
          <div className="form-section">
            <h3>Clinical Measurements</h3>
            <div className="input-row">
              <div className="input-group">
                <label htmlFor="glucose">
                  Glucose (mg/dL) <span className="required">*</span>
                </label>
                <input
                  id="glucose"
                  type="number"
                  name="glucose"
                  value={formData.glucose}
                  onChange={handleInputChange}
                  className={errors.glucose ? 'error' : ''}
                  required
                />
                {errors.glucose && <span className="error-message">{errors.glucose}</span>}
              </div>
              
              <div className="input-group">
                <label htmlFor="blood_pressure">Blood Pressure (mmHg)</label>
                <input
                  id="blood_pressure"
                  type="number"
                  name="blood_pressure"
                  value={formData.blood_pressure}
                  onChange={handleInputChange}
                  className={errors.blood_pressure ? 'error' : ''}
                />
                {errors.blood_pressure && <span className="error-message">{errors.blood_pressure}</span>}
              </div>
            </div>

            <div className="input-row">
              <div className="input-group">
                <label htmlFor="bmi">
                  BMI <span className="required">*</span>
                </label>
                <input
                  id="bmi"
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleInputChange}
                  className={errors.bmi ? 'error' : ''}
                  step="0.1"
                  required
                />
                {errors.bmi && <span className="error-message">{errors.bmi}</span>}
              </div>
              
              <div className="input-group">
                <label htmlFor="hba1c">
                  HbA1c (%) <span className="required">*</span>
                </label>
                <input
                  id="hba1c"
                  type="number"
                  name="hba1c"
                  value={formData.hba1c}
                  onChange={handleInputChange}
                  className={errors.hba1c ? 'error' : ''}
                  step="0.1"
                  required
                />
                {errors.hba1c && <span className="error-message">{errors.hba1c}</span>}
              </div>
            </div>
          </div>

          {/* Lab Values Section */}
          <div className="form-section">
            <h3>Lab Values</h3>
            <div className="input-row">
              <div className="input-group">
                <label htmlFor="insulin">Insulin (μU/mL)</label>
                <input
                  id="insulin"
                  type="number"
                  name="insulin"
                  value={formData.insulin}
                  onChange={handleInputChange}
                  className={errors.insulin ? 'error' : ''}
                  step="0.1"
                />
                {errors.insulin && <span className="error-message">{errors.insulin}</span>}
              </div>
              
              <div className="input-group">
                <label htmlFor="cholesterol">Cholesterol (mg/dL)</label>
                <input
                  id="cholesterol"
                  type="number"
                  name="cholesterol"
                  value={formData.cholesterol}
                  onChange={handleInputChange}
                  className={errors.cholesterol ? 'error' : ''}
                  step="0.1"
                />
                {errors.cholesterol && <span className="error-message">{errors.cholesterol}</span>}
              </div>
            </div>

            <div className="input-row">
              <div className="input-group">
                <label htmlFor="skin_thickness">Skin Thickness (mm)</label>
                <input
                  id="skin_thickness"
                  type="number"
                  name="skin_thickness"
                  value={formData.skin_thickness}
                  onChange={handleInputChange}
                  className={errors.skin_thickness ? 'error' : ''}
                  step="0.1"
                />
                {errors.skin_thickness && <span className="error-message">{errors.skin_thickness}</span>}
              </div>
              
              <div className="input-group">
                <label htmlFor="diabetes_pedigree">Diabetes Pedigree</label>
                <input
                  id="diabetes_pedigree"
                  type="number"
                  name="diabetes_pedigree"
                  value={formData.diabetes_pedigree}
                  onChange={handleInputChange}
                  className={errors.diabetes_pedigree ? 'error' : ''}
                  step="0.001"
                />
                {errors.diabetes_pedigree && <span className="error-message">{errors.diabetes_pedigree}</span>}
              </div>
            </div>
          </div>

          {/* Lifestyle Factors */}
          <div className="form-section">
            <h3>Lifestyle Factors</h3>
            <div className="input-row">
              <div className="input-group">
                <label htmlFor="exercise_weekly">Exercise (days/week)</label>
                <input
                  id="exercise_weekly"
                  type="number"
                  name="exercise_weekly"
                  value={formData.exercise_weekly}
                  onChange={handleInputChange}
                  className={errors.exercise_weekly ? 'error' : ''}
                  min="0"
                  max="7"
                />
                {errors.exercise_weekly && <span className="error-message">{errors.exercise_weekly}</span>}
              </div>
            </div>

            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="smoking"
                  checked={formData.smoking}
                  onChange={handleInputChange}
                />
                <span>Current Smoker</span>
              </label>
              
              <label className="checkbox-label">
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

          {/* Image Upload */}
          <div className="form-section full-width">
            <h3>Retinal Image (Optional)</h3>
            <div className="image-upload-area">
              {!imagePreview ? (
                <label htmlFor="image-upload" className="upload-label">
                  <div className="upload-content">
                    <span className="upload-icon">📷</span>
                    <span>Click to upload retinal image</span>
                    <span className="upload-hint">PNG, JPG up to 10MB</span>
                  </div>
                  <input
                    id="image-upload"
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="file-input"
                  />
                </label>
              ) : (
                <div className="image-preview-container">
                  <img src={imagePreview} alt="Retinal scan preview" className="image-preview" />
                  <button
                    type="button"
                    onClick={() => {
                      setImageFile(null);
                      setImagePreview(null);
                    }}
                    className="btn-remove"
                  >
                    Remove Image
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" className="btn btn-primary btn-large" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Risk'}
          </button>
        </div>
      </form>

      {/* Enhanced Results Display */}
      {prediction && (
        <div className="results-container">
          <h2>Analysis Results</h2>
          
          <div className="risk-summary">
            <div className="risk-card" style={{ borderColor: getRiskColor(prediction.risk_level) }}>
              <h3>Risk Assessment</h3>
              <div className="risk-level" style={{ color: getRiskColor(prediction.risk_level) }}>
                {prediction.risk_level}
              </div>
              <div className="risk-score">
                Score: {(prediction.combined_risk * 100).toFixed(1)}%
              </div>
              <div className="confidence">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </div>
            </div>

            <div className="metrics-card">
              <h3>Risk Components</h3>
              <div className="metric-item">
                <span>Clinical Risk:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${prediction.clinical_risk * 100}%` }}
                  />
                </div>
                <span>{(prediction.clinical_risk * 100).toFixed(1)}%</span>
              </div>
              {prediction.image_risk !== null && (
                <div className="metric-item">
                  <span>Image Risk:</span>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${prediction.image_risk * 100}%` }}
                    />
                  </div>
                  <span>{(prediction.image_risk * 100).toFixed(1)}%</span>
                </div>
              )}
              <div className="metric-item combined-risk">
                <span>Combined Risk:</span>
                <div className="progress-bar">
                  <div 
                    className="progress-fill combined"
                    style={{ width: `${prediction.combined_risk * 100}%` }}
                  />
                </div>
                <span><strong>{(prediction.combined_risk * 100).toFixed(1)}%</strong></span>
              </div>
            </div>
          </div>

          <div className="recommendations-section">
            <h3>Recommendations</h3>
            <ul className="recommendations-list">
              {prediction.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </div>

          {/* Feature Importance */}
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
  );
};

// MetricsSection Component with Charts
const MetricsSection = () => {
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [performanceHistory, setPerformanceHistory] = useState([]);

  useEffect(() => {
    fetchMetrics();
    fetchPerformanceHistory();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('/models/metrics');
      setMetrics(response.data);
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
      // Mock data for demonstration
      setMetrics([
        {
          model_type: 'Clinical Model',
          accuracy: 0.878,
          precision: 0.845,
          recall: 0.892,
          f1_score: 0.868,
          auc_roc: 0.921
        },
        {
          model_type: 'Image Model',
          accuracy: 0.850,
          precision: 0.823,
          recall: 0.876,
          f1_score: 0.849,
          auc_roc: 0.904
        },
        {
          model_type: 'Fusion Model',
          accuracy: 0.920,
          precision: 0.898,
          recall: 0.943,
          f1_score: 0.920,
          auc_roc: 0.957
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const fetchPerformanceHistory = async () => {
    try {
      const response = await axios.get('/models/performance-history');
      setPerformanceHistory(response.data);
    } catch (error) {
      console.error('Failed to fetch performance history:', error);
      // Mock data for demonstration
      setPerformanceHistory([
        { date: '2025-01-01', clinical: 0.860, image: 0.820, fusion: 0.900 },
        { date: '2025-02-01', clinical: 0.865, image: 0.830, fusion: 0.905 },
        { date: '2025-03-01', clinical: 0.870, image: 0.840, fusion: 0.910 },
        { date: '2025-04-01', clinical: 0.875, image: 0.845, fusion: 0.915 },
        { date: '2025-05-01', clinical: 0.878, image: 0.850, fusion: 0.920 }
      ]);
    }
  };

  if (loading) {
    return <div className="loading">Loading metrics...</div>;
  }

  return (
    <div className="metrics-section">
      <div className="section-header">
        <h2>Model Performance Metrics</h2>
        <p>Detailed performance statistics and trends for all deployed models</p>
      </div>
      
      {/* Current Metrics */}
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

      {/* Performance Trends Chart */}
      <div className="chart-section">
        <h3>Performance Trends</h3>
        <div className="chart-container">
          <PerformanceChart performanceHistory={performanceHistory} />
        </div>
      </div>

      {/* Model Comparison */}
      <div className="comparison-section">
        <h3>Model Comparison</h3>
        <div className="comparison-chart">
          {['accuracy', 'precision', 'recall', 'f1_score'].map(metric => (
            <div key={metric} className="comparison-row">
              <div className="metric-name">{metric.replace('_', ' ').toUpperCase()}</div>
              <div className="comparison-bars">
                {metrics.map((model, idx) => (
                  <div key={idx} className="comparison-bar-container">
                    <div className="model-label">{model.model_type}</div>
                    <div className="comparison-bar">
                      <div 
                        className="comparison-fill"
                        style={{ 
                          width: `${model[metric] * 100}%`,
                          backgroundColor: idx === 0 ? '#4CAF50' : idx === 1 ? '#2196F3' : '#FF9800'
                        }}
                      />
                    </div>
                    <div className="comparison-value">{(model[metric] * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Enhanced About Section
const AboutSection = ({ modelInfo }) => {
  return (
    <div className="about-section">
      <div className="section-header">
        <h2>About This System</h2>
      </div>
      
      <div className="about-grid">
        <div className="info-card">
          <h3>System Overview</h3>
          <p>
            This AI-powered system combines clinical data analysis with retinal image 
            processing to detect diabetic retinopathy risk. Using state-of-the-art 
            machine learning models, it provides rapid, accurate risk assessments.
          </p>
        </div>

       

        <div className="info-card">
          <h3>Technology Stack</h3>
          <ul>
            <li><strong>Backend:</strong> FastAPI, Python 3.9+</li>
            <li><strong>ML Models:</strong> XGBoost, Random Forest, ResNet50, VGG16, MobileNetV2</li>
            <li><strong>Image Processing:</strong> TensorFlow, OpenCV, CLAHE preprocessing</li>
            <li><strong>Frontend:</strong> React, Axios</li>
            <li><strong>Deployment:</strong> Docker, Kubernetes-ready</li>
          </ul>
        </div>

        {/* Live Model Information */}
        <div className="info-card">
          <h3>📊 Live Model Information</h3>
          {modelInfo ? (
            <div className="model-info-grid">
              <div className="model-info-item">
                <h4>Clinical Model</h4>
                <p>Status: {modelInfo.clinical_model?.loaded ? '✅ Loaded' : '❌ Not Loaded'}</p>
                <p>Type: {modelInfo.clinical_model?.type}</p>
                <p>Accuracy: {modelInfo.clinical_model?.accuracy ? (modelInfo.clinical_model.accuracy * 100).toFixed(1) + '%' : 'N/A'}</p>
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
          ) : (
            <p>Loading model information...</p>
          )}
        </div>

       
       
        <div className="info-card warning">
          <h3>Important Notice</h3>
          <p>
            This system is for educational and research purposes only. It should not 
            replace professional medical diagnosis. Always consult qualified healthcare 
            providers for medical decisions.
          </p>
        </div>
      </div>
    </div>
  );
};

// Footer Component
const Footer = () => {
  return (
    <footer className="app-footer">
      <div className="footer-content">
      <div className="footer-section">
        <h4>Diabetic Retinopathy Detection</h4>
        <p>AI-powered medical analysis system</p>
      </div>
      
      <div className="footer-section"> 
        <p className="copyright">© 2025 DR Detection System by 💗 Dumindu Thushahn</p>
      </div>
    </div>
      
    </footer>
  );
};

export default App;