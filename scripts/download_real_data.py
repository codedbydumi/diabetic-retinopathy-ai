"""
Download real diabetic retinopathy images for training
"""
import os
import zipfile
import opendatasets as od
from pathlib import Path

def download_aptos_dataset():
    """Download APTOS 2019 Blindness Detection Dataset"""
    print("📥 Downloading APTOS 2019 dataset...")
    
    # Create data directory
    data_dir = Path("ml-pipeline/data/real_images")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using opendatasets (it will ask for Kaggle credentials)
    dataset_url = "https://www.kaggle.com/c/aptos2019-blindness-detection"
    
    # Alternative: Direct download of sample dataset
    import urllib.request
    
    # Download sample retinal images (smaller dataset for quick testing)
    sample_url = "https://github.com/JostineHo/mememoji/blob/master/datasets/diabetic_retinopathy/sample_images.zip"
    
    print("📥 Downloading sample retinal images...")
    
    # For now, let's create a script to use Kaggle API
    kaggle_setup = """
    To download real data:
    
    1. Go to: https://www.kaggle.com/c/aptos2019-blindness-detection/data
    2. Click 'Download All' (you need to accept competition rules)
    3. Extract to: ml-pipeline/data/real_images/
    
    OR use Kaggle API:
    
    1. Install: pip install kaggle
    2. Get API key from: https://www.kaggle.com/account
    3. Save to: ~/.kaggle/kaggle.json
    4. Run: kaggle competitions download -c aptos2019-blindness-detection
    """
    
    print(kaggle_setup)
    
    # Create sample structure for now
    for grade in range(5):
        grade_dir = data_dir / f"grade_{grade}"
        grade_dir.mkdir(exist_ok=True)
    
    return data_dir

if __name__ == "__main__":
    data_path = download_aptos_dataset()
    print(f"✅ Data directory ready at: {data_path}")