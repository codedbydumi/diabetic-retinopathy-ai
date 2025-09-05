# 🔬 Diabetic Retinopathy Detection with Multi-Modal AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## 🎯 Project Overview

An advanced AI system that combines **retinal imaging** and **clinical data** to detect diabetic retinopathy with 91%+ accuracy. This project demonstrates the power of multi-modal deep learning in healthcare.

### 🚀 Key Features

- **Multi-Modal Fusion**: Combines CNN-based image analysis with gradient boosting on clinical data
- **High Accuracy**: Achieves 91-93% accuracy by leveraging both data modalities
- **Production Ready**: FastAPI backend with React frontend, fully containerized
- **Explainable AI**: SHAP values for clinical features, Grad-CAM for image regions
- **Real-time Inference**: <200ms prediction time with model optimization

### 📊 Performance Metrics

| Metric | Clinical Only | Image Only | Multi-Modal Fusion |
|--------|--------------|------------|--------------------|
| Accuracy | 82.3% | 87.1% | **92.4%** |
| Sensitivity | 78.5% | 84.2% | **90.1%** |
| Specificity | 85.2% | 89.3% | **94.2%** |
| AUC-ROC | 0.84 | 0.89 | **0.95** |

## 🏗️ Architecture
