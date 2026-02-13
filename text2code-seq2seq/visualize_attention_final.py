"""
Attention Visualization Script for LSTM+Attention Model
Run after training to visualize attention weights for test examples
"""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import (
    load_and_preprocess_data,
    CodeDataset,
    load_vocab,
    collate_batch,
    set_seed
)
from torch.utils.data import DataLoader
from models.lstm_attention import create_lstm_attention_model
from evaluate_metrics import EvaluationMetrics


class AttentionVisualizer:
    """Visualize attention mechanisms from trained model"""
    
    def __init__(self, vocab_src, vocab_trg):
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.metrics = EvaluationMetrics()
        sns.set_style("whitegrid")
    
    def decode_tokens(self, indices):
        """Convert token indices to string tokens"""
        return self.metrics.decode_tokens(indices, self.vocab_src)
    
    def decode_target(self, indices):
        """Convert target indices to string tokens"""
        return self.metrics.decode_tokens(indices, self.vocab_trg)
    
    def plot_heatmap(self, attention, src_tokens, trg_tokens, 
                     title='Attention Heatmap', save_path=None):
        """
        Plot attention heatmap
        
        Args:
            attention: (trg_len, src_len) attention numpy array
            src_tokens: Source tokens (list)
            trg_tokens: Target tokens (list)
            title: Plot title
            save_path: Path to save figure
        """
        # Convert to numpy if needed
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        # Limit size for readability
        max_src = 25
        max_trg = 25
        
        trg_len = min(len(trg_tokens), attention.shape[0], max_trg)
        src_len = min(len(src_tokens), attention.shape[1], max_src)
        
        attn_trimmed = attention[:trg_len, :src_len]
        
        fig, ax = plt.subplots(figsize=(max(10, src_len * 0.4), max(8, trg_len * 0.4)))
        
        # Create heatmap
        sns.heatmap(attn_trimmed,
                   xticklabels=src_tokens[:src_len],
                   yticklabels=trg_tokens[:trg_len],
                   cmap='YlOrRd',
                   ax=ax,
                   cbar_kws={'label': 'Attention Weight'},
                   vmin=0, vmax=1,
                   linewidths=0.5,
                   annot=False)
        
        ax.set_xlabel('Source Tokens (Docstring)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Generated Tokens (Code)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def generate_with_attention_step_by_step(self, model, src, device, max_len=150):
        """
        Generate code step-by-step to capture attention weights
        
        Args:
            model: Attention seq2seq model
            src: Source tensor (1, src_len)
            device: Device
            max_len: Maximum sequence length
        
        Returns:
            Generated tokens and attention weights matrix
        """
        model.eval()
        
        with torch.no_grad():
            # Encode
            encoder_outputs, (hidden, cell) = model.encoder(src)
            
            # Start decoding
            decoder_input = torch.tensor([[1]], device=device)  # SOS token: shape (1, 1)
            
            generated_tokens = []
            all_attention_weights = []
            
            for step in range(max_len):
                try:
                    # Decoder step with attention
                    # decoder_input: (1, 1), hidden: (1, hidden_dim), cell: (1, hidden_dim)
                    # encoder_outputs: (1, src_len, hidden_dim)
                    decoder_output, (hidden, cell), attn_weights = model.decoder(
                        decoder_input, hidden, cell, encoder_outputs
                    )
                    
                    # decoder_output: (1, 1, vocab_size)
                    # attn_weights: (1, 1, src_len) -> squeeze -> (src_len,)
                    
                    pred = decoder_output.argmax(dim=-1)  # (1, 1)
                    top_token = pred.squeeze().item()
                    
                    generated_tokens.append(top_token)
                    
                    # Store attention weights, squeeze from (1, 1, src_len) to (src_len,)
                    attn = attn_weights.squeeze(0).squeeze(0).cpu().numpy()  # (src_len,)
                    all_attention_weights.append(attn)
                    
                    # Stop if EOS
                    if top_token == 2:  # EOS token idx
                        break
                    
                    decoder_input = torch.tensor([[top_token]], device=device)
                    
                except Exception as e:
                    print(f"Error in generation step {step}: {str(e)}")
                    break
        
        # Stack attention weights: (trg_len, src_len)
        if all_attention_weights:
            attention_matrix = np.array(all_attention_weights)
        else:
            attention_matrix = np.array([])
        
        return generated_tokens, attention_matrix
    
    def visualize_examples(self, model, dataloader, device, num_examples=3, save_dir=None):
        """
        Visualize attention for multiple test examples
        
        Args:
            model: Trained attention model
            dataloader: Test dataloader
            device: Device
            num_examples: Number of examples to visualize
            save_dir: Directory to save visualizations
        """
        model.eval()
        example_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if example_count >= num_examples:
                break
            
            src = batch['docstring'].to(device)
            trg = batch['code'].to(device)
            
            for i in range(src.shape[0]):
                if example_count >= num_examples:
                    break
                
                example_idx = example_count + 1
                print(f"\n{'='*70}")
                print(f"Example {example_idx}")
                print(f"{'='*70}\n")
                
                # Get tokens
                src_tokens = self.decode_tokens(src[i])
                ref_tokens = self.decode_target(trg[i])
                
                # Generate with attention
                gen_tokens, attention = self.generate_with_attention(
                    model, src[i:i+1], device
                )
                
                gen_str = ' '.join([
                    self.vocab_trg.itos.get(idx, '<UNK>')
                    for idx in gen_tokens if idx not in (0, 1, 2)
                ])
                
                # Print
                print(f"Docstring (input):")
                print(f"  {' '.join(src_tokens[:50])}")
                print(f"\nReference (expected):")
                print(f"  {' '.join(ref_tokens[:50])}")
                print(f"\nGenerated (model output):")
                print(f"  {gen_str[:100]}")
                
                # Visualization
                if attention is not None and len(attention) > 0:
                    gen_token_strs = [
                        self.vocab_trg.itos.get(idx, f'<{idx}>')
                        for idx in gen_tokens if idx not in (0, 2)
                    ]
                    
                    # Convert attention to numpy if needed
                    attn_np = attention.cpu().numpy() if isinstance(attention, torch.Tensor) else attention
                    
                    save_path = None
                    if save_dir:
                        save_path = os.path.join(
                            save_dir, 
                            f'attention_example_{example_idx}.png'
                        )
                    
                    self.plot_heatmap(
                        attn_np,
                        src_tokens,
                        gen_token_strs,
                        title=f'Attention Weights - Example {example_idx}',
                        save_path=save_path
                    )
                    
                    # Analyze attention
                    self._analyze_attention(attn_np, src_tokens, gen_token_strs)
                
                example_count += 1
    
    def _analyze_attention(self, attention, src_tokens, trg_tokens):
        """Analyze and interpret attention patterns"""
        print(f"\n📊 Attention Analysis:")
        
        # Convert to numpy if needed
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        # Most attended source tokens (average attention across all target positions)
        mean_attention = attention.mean(axis=0)  # (src_len,)
        top_src_indices = np.argsort(mean_attention)[::-1][:3]
        
        print(f"  Top attended source tokens:")
        for idx in top_src_indices:
            if idx < len(src_tokens):
                weight = mean_attention[idx]
                print(f"    - '{src_tokens[idx]}': {weight:.3f}")
        
        # Attention entropy (measure of focus)
        entropy = -np.sum(attention * np.log(attention + 1e-10), axis=1).mean()
        print(f"  Attention entropy (lower=more focused): {entropy:.3f}")
        
        # Diagonal alignment
        diagonal_score = 0
        count = 0
        for i in range(min(len(trg_tokens), attention.shape[0])):
            expected_pos = int((i / len(trg_tokens)) * attention.shape[1])
            if 0 <= expected_pos < attention.shape[1]:
                diagonal_score += attention[i, expected_pos]
                count += 1
        diagonal_score = diagonal_score / count if count > 0 else 0
        print(f"  Diagonal alignment score: {diagonal_score:.3f}")



def main():
    """Run attention visualization"""
    # Setup
    set_seed(42)
    checkpoint_dir = "/content/drive/MyDrive/text2code-seq2seq/checkpoints"
    output_dir = os.path.join(checkpoint_dir, "attention_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_data, val_data, test_data, docstring_vocab, code_vocab = load_and_preprocess_data(
        num_train=100,
        num_val=100,
        num_test=500,
        max_docstring_len=100,
        max_code_len=150
    )
    
    test_dataset = CodeDataset(test_data, docstring_vocab, code_vocab, 100, 150)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    
    # Load model
    print("Loading model...")
    model = create_lstm_attention_model(
        input_vocab_size=len(docstring_vocab),
        output_vocab_size=len(code_vocab),
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        device=device
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, 'lstm_attention_best.pt')
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"✓ Model loaded from epoch {checkpoint.get('epoch', '?')}")
    
    # Visualize attention
    print("\n" + "="*70)
    print("ATTENTION VISUALIZATION")
    print("="*70)
    
    visualizer = AttentionVisualizer(docstring_vocab, code_vocab)
    visualizer.visualize_examples(
        model, test_loader, device, 
        num_examples=3,
        save_dir=output_dir
    )
    
    print(f"\n{'='*70}")
    print("✓ Attention visualization complete!")
    print(f"  Visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
