"""
Attention Visualization for LSTM+Attention Seq2Seq Model

Visualizes attention weights to understand model behavior:
- Heatmaps showing alignment between docstring and code tokens
- Analysis of what the model attends to
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate_metrics import EvaluationMetrics


class AttentionVisualizer:
    """Visualize and analyze attention weights"""
    
    def __init__(self, vocab_src, vocab_trg):
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.metrics = EvaluationMetrics()
        sns.set_style("whitegrid")
    
    def plot_attention_heatmap(self, attention_weights, src_tokens, trg_tokens, 
                               title='Attention Heatmap', save_path=None,
                               max_src=30, max_trg=30):
        """
        Plot attention heatmap
        
        Args:
            attention_weights: (trg_len, src_len) attention matrix
            src_tokens: Source tokens (list of strings)
            trg_tokens: Generated tokens (list of strings)
            title: Plot title
            save_path: Path to save figure
            max_src: Maximum source tokens to display
            max_trg: Maximum target tokens to display
        """
        # Trim to display size
        trg_len = min(len(trg_tokens), attention_weights.shape[0], max_trg)
        src_len = min(len(src_tokens), attention_weights.shape[1], max_src)
        
        attn = attention_weights[:trg_len, :src_len]
        
        fig, ax = plt.subplots(figsize=(max(10, src_len * 0.4), max(8, trg_len * 0.3)))
        
        sns.heatmap(attn, 
                   xticklabels=src_tokens[:src_len],
                   yticklabels=trg_tokens[:trg_len],
                   cmap='YlOrRd',
                   ax=ax,
                   cbar_kws={'label': 'Attention Weight'},
                   linewidths=0.5)
        
        ax.set_xlabel('Source Tokens (Docstring)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Generated Tokens (Code)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()
    
    def analyze_attention_example(self, model, src_batch, trg_batch, example_idx,
                                  device, has_attention=True):
        """
        Generate and analyze attention for a single example
        
        Args:
            model: Trained model
            src_batch: Source batch
            trg_batch: Target batch
            example_idx: Index of example in batch
            device: Device
            has_attention: Whether model uses attention
        
        Returns:
            dict with example data and attention weights
        """
        model.eval()
        
        with torch.no_grad():
            src = src_batch[example_idx].unsqueeze(0).to(device)
            trg = trg_batch[example_idx].unsqueeze(0).to(device)
            
            # Forward pass
            if has_attention:
                output, attention = model(src, src[0].unsqueeze(0), teacher_forcing_ratio=0)
                # For generation, we need to generate step by step to get attention
                attention = model.generate_with_attention(src, self.vocab_trg, device, max_len=150)
            else:
                output = model(src, trg, teacher_forcing_ratio=0)
                attention = None
            
            # Decode
            src_tokens = self.metrics.decode_tokens(src_batch[example_idx], self.vocab_src)
            ref_tokens = self.metrics.decode_tokens(trg_batch[example_idx], self.vocab_trg)
            pred_tokens = self.metrics.decode_tokens(output.argmax(dim=-1)[0], self.vocab_trg)
            
            return {
                'src_tokens': src_tokens,
                'ref_tokens': ref_tokens,
                'pred_tokens': pred_tokens,
                'attention_weights': attention,
                'output': output
            }
    
    def compute_attention_statistics(self, attention_weights):
        """
        Compute statistics about attention distribution
        
        Args:
            attention_weights: (trg_len, src_len) attention matrix
        
        Returns:
            dict with statistics
        """
        stats = {
            'mean_max_attention': np.mean([row.max() for row in attention_weights]),
            'mean_entropy': np.mean([
                -np.sum(row * np.log(row + 1e-10)) for row in attention_weights
            ]),
            'diagonal_alignment': self._compute_diagonal_alignment(attention_weights),
        }
        return stats
    
    def _compute_diagonal_alignment(self, attention_weights):
        """Measure how much attention follows diagonal alignment"""
        trg_len, src_len = attention_weights.shape
        diagonal_score = 0
        
        for i in range(min(trg_len, src_len)):
            # Expected diagonal position
            expected_pos = int((i / trg_len) * src_len)
            if 0 <= expected_pos < src_len:
                diagonal_score += attention_weights[i, expected_pos]
        
        return diagonal_score / min(trg_len, src_len)
    
    def visualize_multiple_examples(self, model, dataloader, device, num_examples=3,
                                    has_attention=True, save_dir=None):
        """
        Visualize attention for multiple examples
        
        Args:
            model: Trained model
            dataloader: Test dataloader
            device: Device
            num_examples: Number of examples to visualize
            has_attention: Whether model uses attention
            save_dir: Directory to save figures
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
                
                print(f"\n{'='*70}")
                print(f"Example {example_count + 1}")
                print(f"{'='*70}")
                
                # Get example data
                with torch.no_grad():
                    if has_attention:
                        output, attn = model(src[i].unsqueeze(0), trg[i].unsqueeze(0), teacher_forcing_ratio=0)
                    else:
                        output = model(src[i].unsqueeze(0), trg[i].unsqueeze(0), teacher_forcing_ratio=0)
                        attn = None
                
                src_tokens = self.metrics.decode_tokens(src[i], self.vocab_src)
                ref_tokens = self.metrics.decode_tokens(trg[i], self.vocab_trg)
                pred_tokens = self.metrics.decode_tokens(output.argmax(dim=-1)[0], self.vocab_trg)
                
                # Print
                print(f"Docstring: {' '.join(src_tokens[:50])}")
                print(f"Reference: {' '.join(ref_tokens[:50])}")
                print(f"Generated: {' '.join(pred_tokens[:50])}")
                
                # Plot attention if available
                if attn is not None:
                    save_path = None
                    if save_dir:
                        save_path = f"{save_dir}/attention_example_{example_count + 1}.png"
                    
                    self.plot_attention_heatmap(
                        attn,
                        src_tokens,
                        pred_tokens,
                        title=f'Attention Weights - Example {example_count + 1}',
                        save_path=save_path
                    )
                
                example_count += 1


def generate_attention_report(model, test_loader, vocab_src, vocab_trg, device, 
                             num_examples=3, save_dir=None):
    """
    Generate comprehensive attention analysis report
    
    Args:
        model: Trained attention model
        test_loader: Test dataloader
        vocab_src: Source vocabulary
        vocab_trg: Target vocabulary
        device: Device
        num_examples: Number of examples to analyze
        save_dir: Directory to save visualizations
    """
    visualizer = AttentionVisualizer(vocab_src, vocab_trg)
    
    print(f"\n{'='*70}")
    print("ATTENTION ANALYSIS")
    print(f"{'='*70}")
    
    visualizer.visualize_multiple_examples(
        model, test_loader, device, 
        num_examples=num_examples,
        has_attention=True,
        save_dir=save_dir
    )
    
    print(f"\n{'='*70}")
    print("Attention analysis complete!")
    print(f"{'='*70}\n")
