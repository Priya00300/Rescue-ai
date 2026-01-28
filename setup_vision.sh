#!/bin/bash

# RescueAI Vision System - Automated Setup Script
# This script sets up the complete backend infrastructure

set -e  # Exit on error

echo "=========================================="
echo "   RescueAI Vision System Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/8]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# Create directory structure
echo ""
echo -e "${YELLOW}[2/8]${NC} Creating directory structure..."
mkdir -p backend/{models,utils,api,checkpoints,logs}
mkdir -p datasets/{fire,flood,collapse,human}/{train,val}/{images,masks}
mkdir -p datasets/human/{train,val}/annotations
mkdir -p notebooks

echo "✅ Directories created"

# Create virtual environment
echo ""
echo -e "${YELLOW}[3/8]${NC} Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate || source venv/Scripts/activate 2>/dev/null

# Upgrade pip
echo ""
echo -e "${YELLOW}[4/8]${NC} Upgrading pip..."
pip install --upgrade pip --quiet

# Create requirements.txt
echo ""
echo -e "${YELLOW}[5/8]${NC} Creating requirements.txt..."
cat > backend/requirements.txt << 'EOF'
# Core ML
torch>=2.0.0
torchvision>=0.15.0
segmentation-models-pytorch>=0.3.3

# Computer Vision
opencv-python>=4.8.0
albumentations>=1.3.1
Pillow>=10.0.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Utilities
matplotlib>=3.7.0
tqdm>=4.66.0
tensorboard>=2.14.0

# Optional but recommended
timm>=0.9.0
EOF

echo "✅ requirements.txt created"

# Install dependencies
echo ""
echo -e "${YELLOW}[6/8]${NC} Installing dependencies (this may take a few minutes)..."
pip install -r backend/requirements.txt --quiet

echo "✅ Dependencies installed"

# Create sample annotations file
echo ""
echo -e "${YELLOW}[7/8]${NC} Creating sample annotation files..."
cat > datasets/human/train/annotations.json << 'EOF'
{
  "sample_image.jpg": [
    [100, 100, 150, 150],
    [200, 200, 250, 250]
  ]
}
EOF

cat > datasets/human/val/annotations.json << 'EOF'
{
  "sample_image.jpg": [
    [120, 120, 170, 170]
  ]
}
EOF

echo "✅ Sample annotations created"

# Create .gitignore
echo ""
echo -e "${YELLOW}[8/8]${NC} Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Model checkpoints
checkpoints/*.pth
!checkpoints/.gitkeep

# Data
datasets/*
!datasets/README.md

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# TensorBoard
runs/
EOF

# Create checkpoint directory placeholder
touch backend/checkpoints/.gitkeep

echo "✅ .gitignore created"

# Summary
echo ""
echo "=========================================="
echo "   Setup Complete! ✅"
echo "=========================================="
echo ""
echo "📁 Directory Structure:"
echo "   ├── backend/"
echo "   │   ├── models/       (Copy Python files here)"
echo "   │   ├── utils/"
echo "   │   ├── api/"
echo "   │   └── checkpoints/"
echo "   ├── datasets/"
echo "   │   ├── fire/         (Download datasets here)"
echo "   │   ├── flood/"
echo "   │   ├── collapse/"
echo "   │   └── human/"
echo "   └── venv/             (Virtual environment)"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Copy the Python files into backend/ directories:"
echo "   - multi_head_segmentation.py → backend/models/"
echo "   - dataset_loader.py → backend/models/"
echo "   - training.py → backend/models/"
echo "   - preprocessing.py → backend/utils/"
echo "   - grid_converter.py → backend/utils/"
echo "   - server.py → backend/api/"
echo "   - config.py → backend/"
echo "   - train.py → backend/"
echo ""
echo "2. Download datasets (see README for links)"
echo ""
echo "3. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Train the model:"
echo "   cd backend"
echo "   python train.py --mode train --epochs 30"
echo ""
echo "5. Start the API server:"
echo "   python api/server.py"
echo ""
echo "6. Test with:"
echo "   curl http://localhost:8000/"
echo ""
echo "=========================================="

# Create a quick test script
cat > test_setup.sh << 'EOF'
#!/bin/bash
echo "Testing RescueAI Vision Setup..."
echo ""

# Activate venv
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# Test imports
python3 -c "
import torch
import torchvision
import segmentation_models_pytorch as smp
import cv2
import albumentations
import fastapi

print('✅ All core dependencies are installed!')
print('')
print('Package versions:')
print(f'  PyTorch: {torch.__version__}')
print(f'  TorchVision: {torchvision.__version__}')
print(f'  SMP: {smp.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  Albumentations: {albumentations.__version__}')
print(f'  FastAPI: {fastapi.__version__}')
print('')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "Setup test complete! ✅"
EOF

chmod +x test_setup.sh

echo "💡 Tip: Run './test_setup.sh' to verify your installation"
echo ""