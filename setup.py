#!/usr/bin/env python3
"""
Setup script for Cosmetics Brand Analysis project.

This script helps initialize the project environment and validate the setup.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages from requirements.txt."""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def validate_directory_structure():
    """Validate that all necessary directories exist."""
    required_dirs = ['data', 'notebooks', 'src', 'docs', 'images']
    project_root = Path.cwd()
    
    print("📁 Validating directory structure...")
    for directory in required_dirs:
        dir_path = project_root / directory
        if dir_path.exists():
            print(f"  ✅ {directory}/ found")
        else:
            print(f"  ❌ {directory}/ missing")
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ Created {directory}/")
    
    return True

def check_required_files():
    """Check if required files exist."""
    required_files = [
        'README.md',
        'requirements.txt',
        'notebooks/cosmetics_brand_analysis.ipynb',
        'src/utils.py',
        'docs/methodology.md',
        'docs/findings.md'
    ]
    
    print("📄 Checking required files...")
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        return False
    
    return True

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Cosmetics Brand Analysis Configuration

# Data settings
DATA_PATH = "data/"
OUTPUT_PATH = "images/"

# Analysis parameters
MIN_RESPONSE_LENGTH = 10  # Minimum characters for text analysis
NPS_THRESHOLD = 7  # Threshold for promoter classification

# Visualization settings
FIGURE_DPI = 300
PLOT_STYLE = "seaborn-v0_8"
COLOR_PALETTE = "viridis"

# Text analysis settings
MAX_FEATURES = 1000  # Maximum features for TF-IDF
MIN_DF = 2  # Minimum document frequency
STOP_WORDS = "english"

# Export settings
EXPORT_FORMAT = "png"
SAVE_FIGURES = True
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Sample configuration file created: config.py")

def main():
    """Main setup function."""
    print("🚀 Setting up Cosmetics Brand Analysis Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Validate directory structure
    validate_directory_structure()
    
    # Check required files
    check_required_files()
    
    # Install requirements
    if not install_requirements():
        print("\n⚠️  Some packages failed to install. Please install manually:")
        print("pip install -r requirements.txt")
    
    # Create sample config
    create_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your dataset in the data/ folder")
    print("2. Open notebooks/cosmetics_brand_analysis.ipynb in Jupyter")
    print("3. Run the analysis cells step by step")
    print("4. Check the docs/ folder for detailed methodology and findings")
    
    print("\nTo start Jupyter Notebook:")
    print("jupyter notebook")

if __name__ == "__main__":
    main()