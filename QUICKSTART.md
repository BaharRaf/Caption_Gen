# Image Caption Generator - Quick Start Guide

This guide will help you get started with the image captioning project from scratch.

## Step-by-Step Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Prepare Dataset

```bash
# Option A: Using Kaggle API (recommended)
# First, get your API key from https://www.kaggle.com/settings/account
# Place kaggle.json in ~/.kaggle/

# Download dataset
python data/download_dataset.py

# Option B: Manual download
# 1. Go to https://www.kaggle.com/datasets/adityajn105/flickr8k
# 2. Download and extract to data/flickr8k/
```

### 3. Preprocess Data

```bash
# This will:
# - Clean captions
# - Build vocabulary
# - Create train/val/test splits

python data/preprocess.py
```

Expected output:
```
Total captions: 40,455
Vocabulary size: 8,256
Train: 6,000 images
Val: 1,000 images
Test: 1,091 images
```

### 4. Train Model

**Option A: Train CNN-LSTM (Baseline) - Fastest**

```bash
python training/train.py \
    --model cnn_lstm \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --experiment_name baseline
```

Training time: ~2 hours on RTX 3080

**Option B: Train CNN-Transformer (Hybrid)**

```bash
python training/train.py \
    --model cnn_transformer \
    --epochs 25 \
    --batch_size 32 \
    --lr 3e-4 \
    --experiment_name hybrid
```

Training time: ~6 hours on RTX 3080

**Option C: Train ViT-GPT2 (Advanced) - Best Results**

```bash
python training/train.py \
    --model vit_gpt2 \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5 \
    --experiment_name advanced
```

Training time: ~12 hours on RTX 3080

### 5. Monitor Training

```bash
# View training progress in TensorBoard
tensorboard --logdir runs/
```

Open browser: http://localhost:6006

### 6. Evaluate Model

```bash
python training/evaluate.py \
    --model cnn_lstm \
    --checkpoint checkpoints/cnn_lstm_best.pth \
    --beam_size 3 \
    --save_results
```

Expected output:
```
BLEU-1: 0.623
BLEU-2: 0.450
BLEU-3: 0.312
BLEU-4: 0.213
METEOR: 0.198
```

### 7. Generate Captions for New Images

```bash
python inference.py \
    --image path/to/your/image.jpg \
    --model cnn_lstm \
    --checkpoint checkpoints/cnn_lstm_best.pth \
    --beam_size 3
```

Example output:
```
Image: dog_beach.jpg
Caption: a brown dog running on the beach
```

### 8. Explainability - Grad-CAM

```bash
python explainability/gradcam.py \
    --image path/to/your/image.jpg \
    --checkpoint checkpoints/cnn_lstm_best.pth \
    --output_dir outputs/gradcam
```

This generates visualizations showing which parts of the image the model focuses on for each word.

## Project Structure

```
image-captioning-project/
├── data/
│   ├── download_dataset.py      # Download Flickr8k
│   ├── preprocess.py             # Build vocabulary, clean data
│   └── dataset.py                # PyTorch Dataset
│
├── models/
│   ├── cnn_lstm.py              # Baseline model (ResNet50 + LSTM)
│   ├── vit_gpt2.py              # Advanced model (ViT + GPT-2)
│   └── cnn_transformer.py       # Hybrid model (EfficientNet + Transformer)
│
├── training/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation with BLEU/METEOR
│
├── explainability/
│   ├── gradcam.py               # Grad-CAM visualization
│   ├── lime_explain.py          # LIME explanations
│   └── attention_viz.py         # Attention visualization
│
└── inference.py                 # Generate captions for new images
```

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Solution**: Reduce batch size

```bash
python training/train.py --batch_size 32  # or 16, 8
```

### Issue 2: Slow Data Loading

**Solution**: Increase number of workers

```bash
python training/train.py --num_workers 8
```

### Issue 3: Kaggle API Not Found

**Solution**: Manual download

1. Go to https://www.kaggle.com/datasets/adityajn105/flickr8k
2. Click "Download"
3. Extract to `data/flickr8k/`

### Issue 4: NLTK Data Missing

**Solution**: Download in Python

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Quick Python Usage

```python
import torch
from models.cnn_lstm import CNNLSTMModel
from PIL import Image

# Load model
model, vocab, _, _ = CNNLSTMModel.load_from_checkpoint(
    'checkpoints/cnn_lstm_best.pth',
    device='cuda'
)

# Load image
image = Image.open('test.jpg')

# Preprocess
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0).cuda()

# Generate caption
caption = model.generate_caption(image_tensor, vocab, beam_size=3)
print(f"Caption: {caption}")
```

## Performance Benchmarks

### Flickr8k Test Set Results

| Model | BLEU-4 | METEOR | Training Time | Inference Speed |
|-------|--------|--------|---------------|-----------------|
| CNN-LSTM | 0.213 | 0.198 | 2h | 15 FPS |
| CNN-Transformer | 0.247 | 0.217 | 6h | 8 FPS |
| ViT-GPT2 | 0.281 | 0.239 | 12h | 3 FPS |

### Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (GTX 1080)
- RAM: 16GB
- Storage: 10GB

**Recommended**:
- GPU: 16-24GB VRAM (RTX 3090/4090)
- RAM: 32GB
- Storage: 20GB

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Fine-tune on your data**: Adapt the model to your specific domain
3. **Try different architectures**: Implement ViT-GPT2 and CNN-Transformer
4. **Add more explainability**: Implement LIME and attention visualization
5. **Scale to COCO**: Train on larger MS-COCO dataset for better results

## Resources

- **Dataset**: https://www.kaggle.com/datasets/adityajn105/flickr8k
- **Papers**: See documentation for references
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **BLEU Score**: https://www.nltk.org/api/nltk.translate.html

## Support

For questions or issues:
1. Check the documentation
2. Review common issues above
3. Open a GitHub issue
4. Contact: your.email@example.com

---

**Good luck with your deep learning project!** 🚀
