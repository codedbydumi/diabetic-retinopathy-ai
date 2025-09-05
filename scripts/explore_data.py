"""
Data Exploration and Visualization Script
Understand the data before modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self):
        self.data_path = Path("ml-pipeline/data")
        self.processed_path = self.data_path / "processed"
        self.train_path = self.data_path / "train"
        
    def load_data(self):
        """Load all datasets"""
        print("📊 Loading datasets...")
        
        # Load clinical data
        self.clinical_full = pd.read_csv(self.processed_path / "clinical_data_enhanced.csv")
        self.clinical_train = pd.read_csv(self.train_path / "clinical_train.csv")
        
        # Load image metadata
        self.image_metadata = pd.read_csv(self.processed_path / "image_metadata.csv")
        
        print(f"✅ Loaded clinical data: {self.clinical_full.shape}")
        print(f"✅ Loaded image metadata: {self.image_metadata.shape}")
        
    def explore_clinical_data(self):
        """Explore clinical features"""
        print("\n" + "="*60)
        print("🔍 CLINICAL DATA EXPLORATION")
        print("="*60)
        
        # Basic statistics
        print("\n📈 Clinical Features Summary:")
        print(self.clinical_full.describe())
        
        # Check for missing values
        print("\n❓ Missing Values:")
        print(self.clinical_full.isnull().sum())
        
        # Correlation analysis
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.clinical_full.select_dtypes(include=[np.number]).corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Clinical Features Correlation Matrix')
        plt.tight_layout()
        plt.savefig('docs/clinical_correlation.png')
        plt.show()
        
        # Feature distributions
        fig, axes = plt.subplots(4, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        numeric_cols = self.clinical_full.select_dtypes(include=[np.number]).columns
        for idx, col in enumerate(numeric_cols[:16]):
            axes[idx].hist(self.clinical_full[col], bins=30, edgecolor='black')
            axes[idx].set_title(f'{col}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        plt.suptitle('Clinical Features Distribution')
        plt.tight_layout()
        plt.savefig('docs/clinical_distributions.png')
        plt.show()
        
        # Outcome analysis
        outcome_counts = self.clinical_full['outcome'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(outcome_counts.values, labels=['No Diabetes', 'Diabetes'], 
                autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        plt.title('Diabetes Distribution in Dataset')
        plt.savefig('docs/diabetes_distribution.png')
        plt.show()
        
    def explore_image_metadata(self):
        """Explore image metadata"""
        print("\n" + "="*60)
        print("🖼️ IMAGE METADATA EXPLORATION")
        print("="*60)
        
        # DR Grade distribution
        grade_counts = self.image_metadata['dr_grade'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(grade_counts.index, grade_counts.values, 
                       color=['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c'])
        plt.xlabel('Diabetic Retinopathy Grade')
        plt.ylabel('Number of Images')
        plt.title('Distribution of DR Grades in Dataset')
        plt.xticks(range(5), ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.savefig('docs/dr_grade_distribution.png')
        plt.show()
        
        # Quality distribution
        quality_counts = self.image_metadata['quality'].value_counts()
        plt.figure(figsize=(8, 6))
        plt.pie(quality_counts.values, labels=quality_counts.index, 
                autopct='%1.1f%%', startangle=90)
        plt.title('Image Quality Distribution')
        plt.savefig('docs/image_quality_distribution.png')
        plt.show()
        
    def analyze_relationships(self):
        """Analyze relationships between features and outcomes"""
        print("\n" + "="*60)
        print("🔗 FEATURE RELATIONSHIPS")
        print("="*60)
        
        # Top correlated features with outcome
        correlations = self.clinical_full.corr()['outcome'].sort_values(ascending=False)
        print("\n📊 Top Features Correlated with Diabetes:")
        print(correlations[1:11])  # Exclude outcome itself
        
        # Create subplots for top features vs outcome
        top_features = correlations[1:7].index
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features):
            # Create violin plot
            data_no_diabetes = self.clinical_full[self.clinical_full['outcome'] == 0][feature]
            data_diabetes = self.clinical_full[self.clinical_full['outcome'] == 1][feature]
            
            axes[idx].violinplot([data_no_diabetes, data_diabetes], positions=[0, 1])
            axes[idx].set_xticks([0, 1])
            axes[idx].set_xticklabels(['No Diabetes', 'Diabetes'])
            axes[idx].set_ylabel(feature)
            axes[idx].set_title(f'{feature} vs Diabetes Status')
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('Top Features vs Diabetes Status')
        plt.tight_layout()
        plt.savefig('docs/top_features_vs_outcome.png')
        plt.show()
        
    def generate_insights(self):
        """Generate key insights from the data"""
        print("\n" + "="*60)
        print("💡 KEY INSIGHTS")
        print("="*60)
        
        insights = []
        
        # Insight 1: Class imbalance
        diabetes_ratio = self.clinical_full['outcome'].mean()
        insights.append(f"1. Class Distribution: {diabetes_ratio:.1%} have diabetes (mild imbalance)")
        
        # Insight 2: Most predictive features
        correlations = self.clinical_full.corr()['outcome'].abs().sort_values(ascending=False)
        top_feature = correlations.index[1]
        insights.append(f"2. Most Predictive Feature: '{top_feature}' (correlation: {correlations[top_feature]:.3f})")
        
        # Insight 3: Image grade distribution
        severe_ratio = (self.image_metadata['dr_grade'] >= 3).mean()
        insights.append(f"3. Severe Cases: {severe_ratio:.1%} of images show severe/proliferative DR")
        
        # Insight 4: Age analysis
        avg_age_diabetes = self.clinical_full[self.clinical_full['outcome'] == 1]['age'].mean()
        avg_age_no_diabetes = self.clinical_full[self.clinical_full['outcome'] == 0]['age'].mean()
        insights.append(f"4. Age Factor: Diabetic patients are {avg_age_diabetes - avg_age_no_diabetes:.1f} years older on average")
        
        # Insight 5: BMI analysis
        avg_bmi_diabetes = self.clinical_full[self.clinical_full['outcome'] == 1]['bmi'].mean()
        insights.append(f"5. BMI Impact: Average BMI for diabetic patients: {avg_bmi_diabetes:.1f}")
        
        for insight in insights:
            print(f"\n✨ {insight}")
        
        # Save insights
        with open('docs/data_insights.txt', 'w') as f:
            f.write("DATA EXPLORATION INSIGHTS\n")
            f.write("="*50 + "\n\n")
            for insight in insights:
                f.write(f"{insight}\n\n")
        
        print("\n📁 Visualizations saved in 'docs/' folder")
        print("📄 Insights saved to 'docs/data_insights.txt'")

def main():
    print("🔬 Starting Data Exploration")
    print("="*60)
    
    # Create docs directory
    Path("docs").mkdir(exist_ok=True)
    
    explorer = DataExplorer()
    explorer.load_data()
    explorer.explore_clinical_data()
    explorer.explore_image_metadata()
    explorer.analyze_relationships()
    explorer.generate_insights()
    
    print("\n" + "="*60)
    print("✅ Data exploration complete!")
    print("\nNext step: Run 'python ml-pipeline/src/train_clinical.py'")

if __name__ == "__main__":
    main()