"""
Setup Kaggle and download APTOS 2019 dataset
"""
import os
import json
from pathlib import Path
import shutil

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    
    # Create .kaggle directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if kaggle.json exists in current directory
    local_kaggle = Path('kaggle.json')
    
    if local_kaggle.exists():
        # Copy to .kaggle directory
        shutil.copy(local_kaggle, kaggle_dir / 'kaggle.json')
        os.chmod(kaggle_dir / 'kaggle.json', 0o600)
        print("✅ Kaggle credentials configured!")
        return True
    else:
        print("""
        ⚠️ Please setup Kaggle credentials:
        
        1. Go to: https://www.kaggle.com/account
        2. Scroll to 'API' section
        3. Click 'Create New API Token'
        4. Save kaggle.json in this directory
        5. Run this script again
        """)
        return False

if __name__ == "__main__":
    setup_kaggle_credentials()