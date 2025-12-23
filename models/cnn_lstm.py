"""
CNN-LSTM Model for Image Captioning (Baseline)
Encoder: ResNet50
Decoder: 2-layer LSTM
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class Vocabulary:
    """Proper vocabulary with EXPLICIT special token names"""
    
    def __init__(self):
        # FIX: Use explicit token names, NOT empty strings!
        self.word2idx = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3
        }
        self.idx2word = {
            0: '<PAD>',
            1: '<START>',
            2: '<END>',
            3: '<UNK>'
        }
        self.word_count = {}
        self.freq_threshold = 5
    
    def add_word(self, word):
        """Add word to vocabulary"""
        self.word_count[word] = self.word_count.get(word, 0) + 1
    
    def build(self):
        """Build vocabulary with frequency threshold"""
        idx = 4  # Start after special tokens
        for word, count in sorted(self.word_count.items(), key=lambda x: x[1], reverse=True):
            if count >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def __call__(self, word):
        """Get word index, return UNK for unknown words"""
        return self.word2idx.get(word, self.word2idx['<UNK>'])
    
    def __len__(self):
        return len(self.word2idx)


class EncoderCNN(nn.Module):
    """
    CNN Encoder using ResNet50
    FIX: Removed torch.no_grad() to enable gradient flow
    """
    def __init__(self, embed_size=512):
        super(EncoderCNN, self).__init__()
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # FIX #1: Selective freezing instead of torch.no_grad()
        # Freeze early layers for stable transfer learning
        for param in list(self.resnet.parameters())[:100]:
            param.requires_grad = False
        
        # Allow gradients for layer3 and layer4
        for param in list(self.resnet.parameters())[100:]:
            param.requires_grad = True
        
        # Projection layer
        self.embed_size = embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, images):
        """
        Extract feature vectors from images
        FIX: NO torch.no_grad() context - allows gradients to flow!
        """
        # Direct forward pass WITHOUT torch.no_grad()
        features = self.resnet(images)  # (batch, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, 2048)
        features = self.fc(features)  # (batch, embed_size)
        features = self.bn(features)
        features = self.dropout(features)
        return features


class DecoderLSTM(nn.Module):
    """
    LSTM Decoder for caption generation
    Fixed: Proper token handling, temperature sampling, beam search
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderLSTM, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Word embedding layer (padding_idx=0 for <PAD>)
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Temperature for sampling
        self.temperature = 1.0
    
    def forward(self, features, captions, lengths):
        """Generate captions given image features"""
        
        # Embed captions
        embeddings = self.embed(captions)  # (batch, max_len, embed_size)
        
        # Concatenate image features with caption embeddings
        features = features.unsqueeze(1)
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        
        # Pack padded sequence
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward
        hiddens, _ = self.lstm(packed)
        
        # Unpack
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True)
        
        # Apply dropout
        hiddens = self.dropout(hiddens)
        
        # Output layer
        outputs = self.fc(hiddens)  # (batch, max_len, vocab_size)
        
        return outputs
    
    def generate_caption(self, features, vocab, max_length=40, beam_size=3, temperature=0.9):
        """Generate caption using improved search"""
        self.temperature = temperature
        
        if beam_size == 1:
            return self._greedy_search(features, vocab, max_length)
        else:
            return self._beam_search(features, vocab, max_length, beam_size)
    
    def _greedy_search(self, features, vocab, max_length):
        """
        Generate caption using greedy search with temperature sampling
        FIX: Proper explicit token handling
        """
        batch_size = features.size(0)
        captions = []
        
        # Initialize hidden state
        states = None
        inputs = features.unsqueeze(1)  # (batch, 1, embed_size)
        
        # Get proper token indices using EXPLICIT names
        START_IDX = vocab.word2idx.get('<START>', 1)
        END_IDX = vocab.word2idx.get('<END>', 2)
        PAD_IDX = vocab.word2idx.get('<PAD>', 0)
        
        for t in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))  # (batch, vocab_size)
            
            # FIX: Temperature-based sampling instead of argmax
            logits = outputs / self.temperature
            probs = torch.softmax(logits, dim=1)
            
            # Sample from distribution (stochastic, adds diversity)
            predicted = torch.multinomial(probs, 1).squeeze(1)  # (batch,)
            captions.append(predicted.cpu().item())
            
            # Stop if END token generated
            if predicted.item() == END_IDX:
                break
            
            # Prepare next input - embed the predicted word
            inputs = self.embed(predicted.unsqueeze(0))
            inputs = inputs.unsqueeze(1)  # (batch, 1, embed_size)
        
        # Convert indices to words
        # FIX: Properly handle special tokens using explicit names
        caption_words = []
        special_tokens = {START_IDX, END_IDX, PAD_IDX}
        
        for idx in captions:
            if idx not in special_tokens:
                word = vocab.idx2word.get(idx, '<UNK>')
                # Don't add special token names to caption
                if not word.startswith('<'):
                    caption_words.append(word)
        
        return ' '.join(caption_words) if caption_words else "unknown object"
    
    def _beam_search(self, features, vocab, max_length, beam_size):
        """
        Generate caption using beam search with length normalization
        FIX: Proper explicit token handling
        """
        device = features.device
        batch_size = features.size(0)
        
        # Get proper token indices using EXPLICIT names
        START_IDX = vocab.word2idx.get('<START>', 1)
        END_IDX = vocab.word2idx.get('<END>', 2)
        PAD_IDX = vocab.word2idx.get('<PAD>', 0)
        
        # Initialize beam: (sequence, score, states)
        sequences = [[START_IDX]]
        scores = [0.0]
        states = [None]
        
        for step in range(max_length):
            all_candidates = []
            
            for i, (seq, score, state) in enumerate(zip(sequences, scores, states)):
                # If sequence already ended, skip
                if seq[-1] == END_IDX:
                    all_candidates.append((seq, score, state))
                    continue
                
                # Get last token
                last_token = torch.tensor([seq[-1]], device=device)
                
                # Get embedding
                if step == 0:
                    input_emb = features.unsqueeze(1)
                else:
                    input_emb = self.embed(last_token).unsqueeze(0)
                
                # LSTM forward
                with torch.no_grad():
                    hiddens, new_states = self.lstm(input_emb, state)
                    outputs = self.fc(hiddens.squeeze(1))
                
                # Get log probabilities
                log_probs = torch.log_softmax(outputs / self.temperature, dim=1)
                
                # Get top-k
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    
                    # Normalize by length to prevent short sequences
                    normalized_score = new_score / (len(new_seq) ** 0.7)
                    
                    all_candidates.append((new_seq, normalized_score, new_states))
            
            # Select top beam_size candidates
            all_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = [x[0] for x in all_candidates[:beam_size]]
            scores = [x[1] for x in all_candidates[:beam_size]]
            states = [x[2] for x in all_candidates[:beam_size]]
            
            # Stop if all sequences ended
            if all(seq[-1] == END_IDX for seq in sequences):
                break
        
        # Get best sequence
        best_seq = sequences[0]
        
        # Convert to words
        # FIX: Properly handle special tokens using explicit names
        caption_words = []
        special_tokens = {START_IDX, END_IDX, PAD_IDX}
        
        for idx in best_seq:
            if idx not in special_tokens:
                word = vocab.idx2word.get(idx, '<UNK>')
                # Don't add special token names to caption
                if not word.startswith('<'):
                    caption_words.append(word)
        
        return ' '.join(caption_words) if caption_words else "unknown object"


class CNNLSTMModel(nn.Module):
    """Complete CNN-LSTM model for image captioning"""
    
    def __init__(self, embed_size=512, hidden_size=512, vocab_size=8256, num_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, images, captions, lengths):
        """Forward pass"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=40, beam_size=3, temperature=0.9):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            features = self.encoder(image)
            # Unsqueeze if needed
            if features.dim() == 1:
                features = features.unsqueeze(0)
            caption = self.decoder.generate_caption(features, vocab, max_length, beam_size, temperature)
        return caption
    
    def save_checkpoint(self, filepath, vocab, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'vocab': vocab,
            'embed_size': self.decoder.embed_size,
            'hidden_size': self.decoder.hidden_size,
            'vocab_size': self.decoder.vocab_size,
            'num_layers': self.decoder.num_layers,
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch:
            checkpoint['epoch'] = epoch
        if loss:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved to {filepath}")
    
    @classmethod
    def load_from_checkpoint(cls, filepath, device='cuda'):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            embed_size=checkpoint['embed_size'],
            hidden_size=checkpoint['hidden_size'],
            vocab_size=checkpoint['vocab_size'],
            num_layers=checkpoint['num_layers']
        )
        
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model.to(device)
        
        return model, checkpoint.get('vocab'), checkpoint.get('epoch'), checkpoint.get('loss')
