"""
Evaluation script with BLEU, METEOR, CIDEr metrics
"""
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import pickle
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_lstm import CNNLSTMModel
from data.dataset import get_dataloaders

def calculate_bleu_scores(references, hypotheses):
    """Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4"""
    smoothing = SmoothingFunction()
    
    # Tokenize
    references = [[ref.split()] for ref in references]
    hypotheses = [hyp.split() for hyp in hypotheses]
    
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), 
                        smoothing_function=smoothing.method1)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothing.method1)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smoothing.method1)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothing.method1)
    
    return bleu1, bleu2, bleu3, bleu4

def calculate_meteor_score(references, hypotheses):
    """Calculate METEOR score"""
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = meteor_score([ref.split()], hyp.split())
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_model(model, test_loader, vocab, device, beam_size=3, num_samples=None):
    """Evaluate model on test set"""
    model.eval()
    
    all_references = []
    all_hypotheses = []
    
    print(f"\nGenerating captions with beam size {beam_size}...")
    
    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(test_loader)):
            if num_samples and batch_idx * test_loader.batch_size >= num_samples:
                break
            
            images = images.to(device)
            
            # Generate captions for each image in batch
            for i in range(images.size(0)):
                image = images[i:i+1]
                
                # Generate caption
                generated_caption = model.generate_caption(image, vocab, max_length=40, beam_size=beam_size)
                
                # Get reference caption (ground truth)
                caption_indices = captions[i].cpu().numpy()
                reference_words = []
                for idx in caption_indices:
                    word = vocab.idx2word[idx]
                    if word not in ['<START>', '<END>', '<PAD>']:
                        reference_words.append(word)
                reference_caption = ' '.join(reference_words)
                
                all_references.append(reference_caption)
                all_hypotheses.append(generated_caption)
    
    print(f"\nEvaluated {len(all_hypotheses)} captions")
    
    # Calculate metrics
    print("\nCalculating BLEU scores...")
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu_scores(all_references, all_hypotheses)
    
    print("Calculating METEOR score...")
    meteor = calculate_meteor_score(all_references, all_hypotheses)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print("="*70)
    
    # Show some examples
    print("\nSample Predictions:")
    print("="*70)
    for i in range(min(10, len(all_references))):
        print(f"\nImage {i+1}:")
        print(f"Reference: {all_references[i]}")
        print(f"Generated: {all_hypotheses[i]}")
    print("="*70)
    
    return {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': meteor,
        'references': all_references,
        'hypotheses': all_hypotheses
    }

def main(args):
    """Main evaluation function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    if args.model == 'cnn_lstm':
        model, vocab, epoch, loss = CNNLSTMModel.load_from_checkpoint(args.checkpoint, device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"✓ Loaded model from epoch {epoch} with loss {loss:.4f}")
    print(f"✓ Vocabulary size: {len(vocab)}")
    
    # Create test dataloader
    print("\nLoading test data...")
    _, _, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate
    results = evaluate_model(
        model, 
        test_loader, 
        vocab, 
        device, 
        beam_size=args.beam_size,
        num_samples=args.num_samples
    )
    
    # Save results
    if args.save_results:
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics
        results_df = pd.DataFrame([{
            'model': args.model,
            'checkpoint': args.checkpoint,
            'beam_size': args.beam_size,
            'bleu1': results['bleu1'],
            'bleu2': results['bleu2'],
            'bleu3': results['bleu3'],
            'bleu4': results['bleu4'],
            'meteor': results['meteor']
        }])
        results_df.to_csv(output_dir / f'{args.model}_metrics.csv', index=False)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'reference': results['references'],
            'hypothesis': results['hypotheses']
        })
        predictions_df.to_csv(output_dir / f'{args.model}_predictions.csv', index=False)
        
        print(f"\n✓ Results saved to {output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate image captioning model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'vit_gpt2', 'cnn_transformer'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/flickr8k',
                       help='Path to dataset directory')
    parser.add_argument('--vocab_path', type=str, default='data/flickr8k/processed/vocabulary.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Evaluation arguments
    parser.add_argument('--beam_size', type=int, default=3,
                       help='Beam size for caption generation')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    main(args)
