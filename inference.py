"""
Simple inference script for generating captions
"""
import torch
from PIL import Image
from torchvision import transforms
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_lstm import CNNLSTMModel

def generate_caption(model, image_path, vocab, device, beam_size=3, max_length=40):
    """Generate caption for a single image"""
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate caption
    caption = model.generate_caption(image_tensor, vocab, max_length, beam_size)
    
    return caption

def main(args):
    """Main inference function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if args.model == 'cnn_lstm':
        model, vocab, epoch, loss = CNNLSTMModel.load_from_checkpoint(args.checkpoint, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"✓ Loaded {args.model} model")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    print(f"  Vocabulary size: {len(vocab)}\n")
    
    # Process single image or directory
    if Path(args.image).is_file():
        images = [args.image]
    elif Path(args.image).is_dir():
        images = list(Path(args.image).glob('*.jpg')) + list(Path(args.image).glob('*.png'))
    else:
        raise ValueError(f"Invalid image path: {args.image}")
    
    print(f"Processing {len(images)} image(s)...\n")
    print("="*70)
    
    # Generate captions
    for image_path in images:
        caption = generate_caption(model, image_path, vocab, device, args.beam_size, args.max_length)
        print(f"\nImage: {image_path.name}")
        print(f"Caption: {caption}")
        print("-"*70)
    
    print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate captions for images')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'vit_gpt2', 'cnn_transformer'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--beam_size', type=int, default=3,
                       help='Beam size for caption generation')
    parser.add_argument('--max_length', type=int, default=40,
                       help='Maximum caption length')
    
    args = parser.parse_args()
    
    main(args)
