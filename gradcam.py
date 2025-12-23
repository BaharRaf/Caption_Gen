"""
Grad-CAM visualization for image captioning
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.cnn_lstm import CNNLSTMModel
from torchvision import transforms

class GradCAM:
    """Grad-CAM for CNN encoder"""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.model.encoder.resnet[-2].register_forward_hook(self.save_activation)
        self.model.encoder.resnet[-2].register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, image, caption_score):
        """Generate CAM heatmap"""
        # Backward pass
        self.model.zero_grad()
        caption_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Calculate weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET):
    """Apply heatmap on image"""
    # Resize activation map to image size
    h, w = org_img.shape[:2]
    activation_map = cv2.resize(activation_map, (w, h))
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed_img = heatmap * 0.4 + org_img * 0.6
    overlayed_img = overlayed_img / overlayed_img.max()
    
    return overlayed_img

def visualize_gradcam(model, image_path, vocab, device, output_dir='outputs/gradcam'):
    """Generate and save Grad-CAM visualizations"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create GradCAM
    gradcam = GradCAM(model)
    
    # Generate caption and get features
    model.eval()
    with torch.enable_grad():
        features = model.encoder(image_tensor)
        
        # Generate first few words
        inputs = features.unsqueeze(1)
        states = None
        generated_words = []
        
        for step in range(10):  # Generate 10 words
            hiddens, states = model.decoder.lstm(inputs, states)
            outputs = model.decoder.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            
            word_idx = predicted.item()
            word = vocab.idx2word[word_idx]
            
            if word == '<END>':
                break
            
            if word not in ['<START>', '<PAD>']:
                generated_words.append(word)
                
                # Generate CAM for this word
                cam = gradcam.generate_cam(image_tensor, outputs[0, word_idx])
                
                # Visualize
                original_resized = cv2.resize(original_image, (224, 224))
                overlayed = apply_colormap_on_image(original_resized, cam)
                
                # Save visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(original_image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title(f'Activation Map for "{word}"')
                axes[1].axis('off')
                
                axes[2].imshow(overlayed)
                axes[2].set_title(f'Grad-CAM Overlay for "{word}"')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'gradcam_word_{step}_{word}.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # Prepare next input
            inputs = model.decoder.embed(predicted).unsqueeze(1)
    
    caption = ' '.join(generated_words)
    print(f"\nGenerated caption: {caption}")
    print(f"✓ Grad-CAM visualizations saved to {output_dir}/")
    
    return caption

def main(args):
    """Main function"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, vocab, _, _ = CNNLSTMModel.load_from_checkpoint(args.checkpoint, device)
    print("✓ Model loaded")
    
    # Generate Grad-CAM
    print(f"\nGenerating Grad-CAM for {args.image}...")
    caption = visualize_gradcam(model, args.image, vocab, device, args.output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations')
    
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='outputs/gradcam',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    main(args)
