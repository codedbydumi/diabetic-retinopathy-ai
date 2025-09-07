import React from 'react';
import './LoadingStates.css';

export const PredictionLoading = () => (
  <div className="prediction-loading">
    <div className="loading-animation">
      <div className="scanner">
        <div className="scan-line"></div>
      </div>
    </div>
    <h3>Analyzing Medical Data...</h3>
    <div className="loading-steps">
      <div className="step active">
        <span className="step-icon">✓</span>
        <span>Processing clinical data</span>
      </div>
      <div className="step active">
        <span className="step-icon">⟳</span>
        <span>Analyzing retinal image</span>
      </div>
      <div className="step">
        <span className="step-icon">○</span>
        <span>Generating risk assessment</span>
      </div>
    </div>
  </div>
);

export const SkeletonLoader = () => (
  <div className="skeleton-loader">
    <div className="skeleton-header"></div>
    <div className="skeleton-text"></div>
    <div className="skeleton-text short"></div>
  </div>
);