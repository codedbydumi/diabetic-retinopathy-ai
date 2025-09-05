"""
Data Download and Preparation Script
This script downloads and prepares both image and clinical datasets
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import zipfile
import json
from tqdm import tqdm
import shutil

class DataDownloader:
    def __init__(self):
        self.base_path = Path("ml-pipeline/data")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'raw': self.base_path / 'raw',
            'processed': self.base_path / 'processed',
            'train': self.base_path / 'train',
            'val': self.base_path / 'val',
            'test': self.base_path / 'test'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_clinical_data(self):
        """Download PIMA Indians Diabetes Dataset"""
        print("📊 Downloading clinical diabetes data...")
        
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        columns = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
        ]
        
        # Download data
        filepath = self.dirs['raw'] / 'diabetes_clinical.csv'
        urllib.request.urlretrieve(url, filepath)
        
        # Load and add column names
        df = pd.read_csv(filepath, header=None, names=columns)
        
        # Add synthetic features to make it more realistic
        np.random.seed(42)
        df['hba1c'] = np.random.normal(5.7 + df['outcome'] * 1.2, 0.8, len(df))
        df['cholesterol'] = np.random.normal(200 + df['outcome'] * 20, 30, len(df))
        df['smoking'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        df['family_history'] = np.random.choice([0, 1], size=len(df), p=[0.6, 0.4])
        df['exercise_weekly'] = np.random.randint(0, 8, len(df))
        df['patient_id'] = [f"P{str(i).zfill(5)}" for i in range(len(df))]
        
        # Save enhanced dataset
        enhanced_path = self.dirs['processed'] / 'clinical_data_enhanced.csv'
        df.to_csv(enhanced_path, index=False)
        
        print(f"✅ Clinical data saved: {enhanced_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Diabetic patients: {df['outcome'].sum()} / {len(df)}")
        
        return df
    
    def generate_synthetic_images_metadata(self):
        """
        Generate metadata for synthetic retinal images
        Since we can't download actual medical images directly,
        we'll create metadata and use sample images
        """
        print("\n🖼️ Generating retinal image metadata...")
        
        # Create synthetic metadata for 1000 patients
        n_samples = 1000
        np.random.seed(42)
        
        # Diabetic Retinopathy grades (0-4)
        # 0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative
        grades_distribution = [0.5, 0.25, 0.15, 0.07, 0.03]
        
        image_metadata = []
        
        for i in range(n_samples):
            grade = np.random.choice([0, 1, 2, 3, 4], p=grades_distribution)
            
            metadata = {
                'image_id': f"IMG_{str(i).zfill(5)}",
                'patient_id': f"P{str(np.random.randint(0, 768)).zfill(5)}",  # Link to clinical data
                'dr_grade': grade,
                'quality': np.random.choice(['Good', 'Fair', 'Poor'], p=[0.7, 0.25, 0.05]),
                'eye': np.random.choice(['Left', 'Right']),
                'image_width': 1024,
                'image_height': 1024,
                'has_macular_edema': np.random.choice([0, 1], p=[0.9, 0.1] if grade < 2 else [0.7, 0.3]),
                'exam_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            }
            image_metadata.append(metadata)
        
        df_images = pd.DataFrame(image_metadata)
        
        # Save metadata
        metadata_path = self.dirs['processed'] / 'image_metadata.csv'
        df_images.to_csv(metadata_path, index=False)
        
        print(f"✅ Image metadata saved: {metadata_path}")
        print(f"   Total images: {len(df_images)}")
        print(f"   Grade distribution:")
        print(df_images['dr_grade'].value_counts().sort_index())
        
        # Create sample image directories
        for split in ['train', 'val', 'test']:
            for grade in range(5):
                grade_dir = self.dirs[split] / f'grade_{grade}'
                grade_dir.mkdir(parents=True, exist_ok=True)
        
        return df_images
    
    def split_data(self, clinical_df, image_df):
        """Split data into train/val/test sets"""
        print("\n🔄 Splitting data into train/val/test...")
        
        from sklearn.model_selection import train_test_split
        
        # Split clinical data
        X = clinical_df.drop(['outcome', 'patient_id'], axis=1)
        y = clinical_df['outcome']
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # Save splits
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        for split_name, (X_split, y_split) in splits.items():
            split_df = pd.concat([X_split, y_split], axis=1)
            split_path = self.dirs[split_name] / f'clinical_{split_name}.csv'
            split_df.to_csv(split_path, index=False)
            print(f"   {split_name}: {len(split_df)} samples saved")
        
        # Split image metadata
        image_train, image_temp = train_test_split(
            image_df, test_size=0.3, random_state=42, stratify=image_df['dr_grade']
        )
        image_val, image_test = train_test_split(
            image_temp, test_size=0.5, random_state=42, stratify=image_temp['dr_grade']
        )
        
        # Save image splits
        image_splits = {
            'train': image_train,
            'val': image_val,
            'test': image_test
        }
        
        for split_name, split_df in image_splits.items():
            split_path = self.dirs[split_name] / f'image_metadata_{split_name}.csv'
            split_df.to_csv(split_path, index=False)
            print(f"   {split_name} images: {len(split_df)} samples saved")
        
        print("\n✅ Data splitting complete!")
    
    def create_dataset_info(self):
        """Create dataset information file"""
        info = {
            "dataset_name": "Diabetic Retinopathy Multi-Modal Dataset",
            "version": "1.0.0",
            "description": "Combined retinal imaging and clinical data for DR detection",
            "clinical_features": [
                "pregnancies", "glucose", "blood_pressure", "skin_thickness",
                "insulin", "bmi", "diabetes_pedigree", "age", "hba1c",
                "cholesterol", "smoking", "family_history", "exercise_weekly"
            ],
            "image_grades": {
                "0": "No DR",
                "1": "Mild DR",
                "2": "Moderate DR",
                "3": "Severe DR",
                "4": "Proliferative DR"
            },
            "splits": {
                "train": 0.6,
                "validation": 0.2,
                "test": 0.2
            }
        }
        
        info_path = self.base_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📄 Dataset info saved: {info_path}")

def main():
    print("🚀 Starting Data Download and Preparation")
    print("=" * 50)
    
    downloader = DataDownloader()
    
    # Download and prepare data
    clinical_df = downloader.download_clinical_data()
    image_df = downloader.generate_synthetic_images_metadata()
    
    # Split data
    downloader.split_data(clinical_df, image_df)
    
    # Create dataset info
    downloader.create_dataset_info()
    
    print("\n" + "=" * 50)
    print("✨ Data preparation complete!")
    print("\nNext steps:")
    print("1. Run 'python scripts/explore_data.py' to explore the data")
    print("2. Run 'python ml-pipeline/src/train_clinical.py' to train clinical model")
    print("3. Run 'python ml-pipeline/src/train_image.py' to train image model")

if __name__ == "__main__":
    main()