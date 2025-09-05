"""
Clinical Model Training Script
Train XGBoost and Random Forest models on clinical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ClinicalModelTrainer:
    def __init__(self):
        self.data_path = Path("ml-pipeline/data")
        self.model_path = Path("ml-pipeline/models")
        self.model_path.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load training and validation data"""
        print("📊 Loading clinical data...")
        
        # Load train data
        train_df = pd.read_csv(self.data_path / "train" / "clinical_train.csv")
        val_df = pd.read_csv(self.data_path / "val" / "clinical_val.csv")
        test_df = pd.read_csv(self.data_path / "test" / "clinical_test.csv")
        
        # Separate features and targets
        self.X_train = train_df.drop('outcome', axis=1)
        self.y_train = train_df['outcome']
        
        self.X_val = val_df.drop('outcome', axis=1)
        self.y_val = val_df['outcome']
        
        self.X_test = test_df.drop('outcome', axis=1)
        self.y_test = test_df['outcome']
        
        print(f"✅ Train set: {self.X_train.shape}")
        print(f"✅ Validation set: {self.X_val.shape}")
        print(f"✅ Test set: {self.X_test.shape}")
        
    def preprocess_data(self):
        """Scale features"""
        print("\n🔧 Preprocessing data...")
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.model_path / 'clinical_scaler.pkl')
        print("✅ Scaler saved")
        
    def train_xgboost(self):
        """Train XGBoost model with hyperparameter tuning"""
        print("\n🚀 Training XGBoost model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        # Create base model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Grid search with cross-validation
        print("   Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Best model
        self.models['xgboost'] = grid_search.best_estimator_
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on validation set
        y_pred = self.models['xgboost'].predict(self.X_val_scaled)
        y_pred_proba = self.models['xgboost'].predict_proba(self.X_val_scaled)[:, 1]
        
        self.results['xgboost'] = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'auc_roc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        print(f"✅ XGBoost Validation Accuracy: {self.results['xgboost']['accuracy']:.4f}")
        print(f"   AUC-ROC: {self.results['xgboost']['auc_roc']:.4f}")
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Create base model
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        print("   Performing hyperparameter tuning...")
        grid_search = GridSearchCV(
            rf_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        # Best model
        self.models['random_forest'] = grid_search.best_estimator_
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate
        y_pred = self.models['random_forest'].predict(self.X_val_scaled)
        y_pred_proba = self.models['random_forest'].predict_proba(self.X_val_scaled)[:, 1]
        
        self.results['random_forest'] = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'auc_roc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        print(f"✅ Random Forest Validation Accuracy: {self.results['random_forest']['accuracy']:.4f}")
        print(f"   AUC-ROC: {self.results['random_forest']['auc_roc']:.4f}")
        
    def evaluate_models(self):
        """Comprehensive evaluation of both models"""
        print("\n📊 Model Evaluation on Test Set")
        print("="*50)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Calculate metrics
            test_results = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'auc_roc': roc_auc_score(self.y_test, y_pred_proba)
            }
            
            print(f"\n{model_name.upper()} Results:")
            print(f"  Accuracy:  {test_results['accuracy']:.4f}")
            print(f"  Precision: {test_results['precision']:.4f}")
            print(f"  Recall:    {test_results['recall']:.4f}")
            print(f"  F1-Score:  {test_results['f1']:.4f}")
            print(f"  AUC-ROC:   {test_results['auc_roc']:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx, 0], cbar=False)
            axes[idx, 0].set_title(f'{model_name} - Confusion Matrix')
            axes[idx, 0].set_xlabel('Predicted')
            axes[idx, 0].set_ylabel('Actual')
            
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:  # XGBoost
                importance = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            axes[idx, 1].barh(feature_importance['feature'], 
                            feature_importance['importance'])
            axes[idx, 1].set_xlabel('Importance')
            axes[idx, 1].set_title(f'{model_name} - Top 10 Features')
            axes[idx, 1].invert_yaxis()
            
            # Save test results
            self.results[f'{model_name}_test'] = test_results
        
        plt.tight_layout()
        plt.savefig('docs/clinical_model_evaluation.png')
        plt.show()
        
    def save_models(self):
        """Save trained models and results"""
        print("\n💾 Saving models...")
        
        # Save models
        for model_name, model in self.models.items():
            model_file = self.model_path / f'clinical_{model_name}.pkl'
            joblib.dump(model, model_file)
            print(f"   ✅ {model_name} saved to {model_file}")
        
        # Save results
        results_file = self.model_path / 'clinical_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'features': list(self.X_train.columns),
            'n_features': len(self.X_train.columns),
            'n_train_samples': len(self.X_train),
            'n_val_samples': len(self.X_val),
            'n_test_samples': len(self.X_test),
            'best_model': max(self.results, key=lambda x: self.results[x]['accuracy']),
            'models': list(self.models.keys())
        }
        
        with open(self.model_path / 'clinical_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Results saved to {results_file}")
        print(f"   ✅ Metadata saved")
        
    def create_ensemble(self):
        """Create an ensemble of both models"""
        print("\n🎯 Creating Ensemble Model...")
        
        # Simple voting ensemble
        from sklearn.ensemble import VotingClassifier
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft'  # Use probability predictions
        )
        
        # Fit on full training data
        ensemble.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        y_pred = ensemble.predict(self.X_test_scaled)
        y_pred_proba = ensemble.predict_proba(self.X_test_scaled)[:, 1]
        
        ensemble_results = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'auc_roc': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        print(f"✅ Ensemble Test Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"   AUC-ROC: {ensemble_results['auc_roc']:.4f}")
        
        # Save ensemble
        joblib.dump(ensemble, self.model_path / 'clinical_ensemble.pkl')
        self.results['ensemble_test'] = ensemble_results
        
        # Update results file
        with open(self.model_path / 'clinical_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

def main():
    print("🏥 Clinical Model Training Pipeline")
    print("="*50)
    
    trainer = ClinicalModelTrainer()
    
    # Pipeline
    trainer.load_data()
    trainer.preprocess_data()
    trainer.train_xgboost()
    trainer.train_random_forest()
    trainer.evaluate_models()
    trainer.create_ensemble()
    trainer.save_models()
    
    print("\n" + "="*50)
    print("✨ Clinical model training complete!")
    print(f"\n📊 Best Model: {max(trainer.results, key=lambda x: trainer.results[x]['accuracy'])}")
    print(f"   Accuracy: {max(trainer.results.values(), key=lambda x: x['accuracy'])['accuracy']:.4f}")
    print("\nNext step: Run 'python ml-pipeline/src/train_image.py'")

if __name__ == "__main__":
    main()