#!/bin/bash

# Image Caption Generator - Automated Setup Script
# This script automates the entire setup process

echo "======================================================================"
echo "Image Caption Generator - Automated Setup"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Check Python installation
echo "Step 1: Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo ""

# Step 2: Create virtual environment
echo "Step 2: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Step 4: Install dependencies
echo "Step 4: Installing dependencies..."
echo "This may take several minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_success "Dependencies installed"
echo ""

# Step 5: Download NLTK data
echo "Step 5: Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"
print_success "NLTK data downloaded"
echo ""

# Step 6: Create directory structure
echo "Step 6: Creating directory structure..."
mkdir -p data/flickr8k/Images
mkdir -p data/flickr8k/processed
mkdir -p checkpoints
mkdir -p outputs/gradcam
mkdir -p outputs/lime
mkdir -p outputs/attention
mkdir -p results
mkdir -p runs
print_success "Directories created"
echo ""

# Step 7: Check for dataset
echo "Step 7: Checking for dataset..."
if [ -f "data/flickr8k/captions.txt" ]; then
    print_success "Dataset found"
    NUM_IMAGES=$(ls -1 data/flickr8k/Images/*.jpg 2>/dev/null | wc -l)
    echo "  Images found: $NUM_IMAGES"
else
    print_warning "Dataset not found"
    echo ""
    echo "Please download the dataset:"
    echo "  Option A: python data/download_dataset.py (requires Kaggle API)"
    echo "  Option B: Manual download from https://www.kaggle.com/datasets/adityajn105/flickr8k"
    echo ""
    read -p "Do you want to try downloading now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python data/download_dataset.py
    else
        print_warning "Skipping dataset download. You can run it later."
    fi
fi
echo ""

# Step 8: Preprocess data
echo "Step 8: Preprocessing data..."
if [ -f "data/flickr8k/captions.txt" ]; then
    if [ -f "data/flickr8k/processed/vocabulary.pkl" ]; then
        print_warning "Preprocessed data already exists"
        read -p "Do you want to preprocess again? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python data/preprocess.py
            print_success "Data preprocessed"
        fi
    else
        python data/preprocess.py
        print_success "Data preprocessed"
    fi
else
    print_warning "Cannot preprocess without dataset. Skipping."
fi
echo ""

# Step 9: Summary
echo "======================================================================"
echo "Setup Summary"
echo "======================================================================"
echo ""
echo "Installation complete! Here's what you can do next:"
echo ""
echo "1. Train a model:"
echo "   python training/train.py --model cnn_lstm --epochs 20"
echo ""
echo "2. Evaluate a model:"
echo "   python training/evaluate.py --model cnn_lstm --checkpoint checkpoints/cnn_lstm_best.pth"
echo ""
echo "3. Generate captions:"
echo "   python inference.py --image path/to/image.jpg --model cnn_lstm --checkpoint checkpoints/cnn_lstm_best.pth"
echo ""
echo "4. Create Grad-CAM visualizations:"
echo "   python explainability/gradcam.py --image path/to/image.jpg --checkpoint checkpoints/cnn_lstm_best.pth"
echo ""
echo "5. Monitor training with TensorBoard:"
echo "   tensorboard --logdir runs/"
echo ""
echo "For detailed instructions, see QUICKSTART.md"
echo ""
echo "======================================================================"
echo "Ready to start your image captioning project! 🚀"
echo "======================================================================"
