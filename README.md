# Image Caption Generator - Deep Learning Project

Complete implementation of image captioning system with three architectures: CNN-LSTM, Vision Transformer + GPT-2, and CNN + Transformer with Attention.

## Project Structure

```
image-captioning-project/
├── data/
│   ├── download_dataset.py      # Download Flickr8k from Kaggle
│   ├── preprocess.py             # Preprocess images and captions
│   └── dataset.py                # PyTorch Dataset class
├── models/
│   ├── cnn_lstm.py              # Baseline CNN-LSTM model
│   ├── vit_gpt2.py              # Vision Transformer + GPT-2
│   ├── cnn_transformer.py       # CNN + Transformer with Attention
│   └── utils.py                 # Model utilities
├── training/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── config.py                # Configuration
├── explainability/
│   ├── gradcam.py               # Grad-CAM visualization
│   ├── lime_explain.py          # LIME explanations
│   └── attention_viz.py         # Attention visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_models.ipynb
│   └── 03_evaluation_analysis.ipynb
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### 3. Download Flickr8k Dataset

**Option A: Using Kaggle API**

```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle API token (place kaggle.json in ~/.kaggle/)
# Download from: https://www.kaggle.com/settings/account

# Download dataset
python data/download_dataset.py
```

**Option B: Manual Download**

1. Go to https://www.kaggle.com/datasets/adityajn105/flickr8k
2. Download and extract to `data/flickr8k/`

Dataset structure:
```
data/flickr8k/
├── Images/           # 8,091 JPEG images
└── captions.txt      # Image-caption pairs
```

## Quick Start

### Train All Models

```bash
# Train CNN-LSTM (baseline)
python training/train.py --model cnn_lstm --epochs 20 --batch_size 64

# Train ViT-GPT2 (advanced)
python training/train.py --model vit_gpt2 --epochs 15 --batch_size 16

# Train CNN-Transformer (hybrid)
python training/train.py --model cnn_transformer --epochs 25 --batch_size 32
```

### Evaluate Models

```bash
python training/evaluate.py --model cnn_lstm --checkpoint checkpoints/cnn_lstm_best.pth
```

### Generate Captions

```python
from models.cnn_lstm import CNNLSTMModel
from PIL import Image
import torch

# Load model
model = CNNLSTMModel.load_from_checkpoint('checkpoints/cnn_lstm_best.pth')
model.eval()

# Load image
image = Image.open('test_image.jpg')

# Generate caption
caption = model.generate_caption(image, max_length=40, beam_size=3)
print(f"Caption: {caption}")
```

### Explainability

```bash
# Generate Grad-CAM visualizations
python explainability/gradcam.py --image test_image.jpg --model cnn_lstm

# Generate LIME explanations
python explainability/lime_explain.py --image test_image.jpg --model vit_gpt2

# Visualize attention
python explainability/attention_viz.py --image test_image.jpg --model cnn_transformer
```

## Model Architectures

### 1. CNN-LSTM (Baseline)
- **Encoder**: ResNet50 (pre-trained on ImageNet)
- **Decoder**: 2-layer LSTM
- **Parameters**: ~28M
- **Training Time**: ~2 hours
- **Expected BLEU-4**: 0.21

### 2. Vision Transformer + GPT-2 (Advanced)
- **Encoder**: ViT-B/16 (pre-trained)
- **Decoder**: GPT-2 Small with cross-attention
- **Parameters**: ~124M
- **Training Time**: ~12 hours
- **Expected BLEU-4**: 0.28

### 3. CNN + Transformer with Attention (Hybrid)
- **Encoder**: EfficientNet-B3
- **Decoder**: 6-layer Transformer with Bahdanau attention
- **Parameters**: ~52M
- **Training Time**: ~6 hours
- **Expected BLEU-4**: 0.25

## Training Configuration

### Hardware Requirements
- **Minimum**: GPU with 8GB VRAM (GTX 1080 or better)
- **Recommended**: GPU with 16-24GB VRAM (RTX 3090/4090, A5000)
- **RAM**: 16GB minimum, 32GB recommended

### Hyperparameters (Default)

| Model | Learning Rate | Batch Size | Epochs | Optimizer |
|-------|--------------|------------|--------|-----------|
| CNN-LSTM | 1e-4 | 64 | 20 | Adam |
| ViT-GPT2 | 5e-5 | 16 | 15 | AdamW |
| CNN-Trans | 3e-4 | 32 | 25 | Adam |

## Evaluation Metrics

The project implements multiple evaluation metrics:

- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap with references
- **METEOR**: Considers synonyms and word order
- **CIDEr**: Consensus-based metric (TF-IDF weighted)
- **ROUGE-L**: Longest common subsequence

## Project Files Description

### Data Processing
- `data/download_dataset.py`: Downloads Flickr8k from Kaggle
- `data/preprocess.py`: Preprocesses images and builds vocabulary
- `data/dataset.py`: PyTorch Dataset and DataLoader

### Models
- `models/cnn_lstm.py`: CNN-LSTM encoder-decoder
- `models/vit_gpt2.py`: Vision Transformer + GPT-2
- `models/cnn_transformer.py`: CNN + Transformer with attention
- `models/utils.py`: Beam search, caption generation utilities

### Training
- `training/train.py`: Main training loop for all models
- `training/evaluate.py`: Evaluation on test set
- `training/config.py`: Configuration management

### Explainability
- `explainability/gradcam.py`: Grad-CAM saliency maps
- `explainability/lime_explain.py`: LIME local explanations
- `explainability/attention_viz.py`: Attention weight visualization

### Notebooks
- `notebooks/01_data_exploration.ipynb`: EDA and statistics
- `notebooks/02_train_models.ipynb`: Interactive training
- `notebooks/03_evaluation_analysis.ipynb`: Results analysis

## Expected Results

### Flickr8k Test Set Performance

| Model | BLEU-1 | BLEU-4 | METEOR | CIDEr |
|-------|--------|--------|--------|-------|
| CNN-LSTM | 0.623 | 0.213 | 0.198 | 0.512 |
| ViT-GPT2 | **0.687** | **0.281** | **0.239** | **0.642** |
| CNN-Trans | 0.651 | 0.247 | 0.217 | 0.578 |

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training (`--fp16`)

### Slow Training
- Use multiple workers for DataLoader
- Enable mixed precision
- Cache preprocessed features

### Poor Results
- Check data preprocessing (image normalization)
- Verify vocabulary size and coverage
- Ensure pre-trained weights are loaded correctly
- Try different learning rates

## Citations

If you use this code, please cite:

```bibtex
@misc{image_caption_generator_2025,
  title={Image Caption Generator: Deep Learning Project},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/image-captioning}}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Flickr8k dataset: Hodosh et al. (2013)
- PyTorch and Hugging Face Transformers
- Research papers cited in documentation

## Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com

---

**Note**: This is an academic project for Deep Learning course. Expected training time: 20-30 hours total for all models.