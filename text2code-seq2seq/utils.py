"""
Utility Functions for Text-to-Code Generation
Helper functions for metrics, visualization, and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import ast
import json


# Import constants from data_preprocessing
try:
    from data_preprocessing import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
except ImportError:
    # Fallback values if import fails
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3


def decode_sequence(indices, vocab):
    """
    Convert indices to tokens, removing special tokens
    
    Args:
        indices: List or tensor of token indices
        vocab: Vocabulary object with itos (index to string) mapping
        
    Returns:
        List of tokens as strings
    """
    tokens = []
    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        if idx == EOS_IDX or idx == PAD_IDX:
            break
        if idx == SOS_IDX:
            continue
            
        token = vocab.itos.get(idx, '<UNK>')
        tokens.append(token)
    
    return tokens


def indices_to_sentence(indices, vocab):
    """
    Convert indices to sentence string
    
    Args:
        indices: List or tensor of token indices
        vocab: Vocabulary object
        
    Returns:
        String representation of the sentence
    """
    tokens = decode_sequence(indices, vocab)
    return ' '.join(tokens)


def prepare_test_example(text, vocab, device='cpu', max_len=50):
    """
    Prepare a test example for model input
    
    Args:
        text: Input text string
        vocab: Vocabulary object
        device: torch device
        max_len: Maximum sequence length
        
    Returns:
        Tensor of shape [1, seq_len]
    """
    # Tokenize
    tokens = text.lower().split()
    
    # Convert to indices
    indices = [SOS_IDX]
    for token in tokens[:max_len-2]:  # Leave room for SOS and EOS
        idx = vocab.stoi.get(token, UNK_IDX)
        indices.append(idx)
    indices.append(EOS_IDX)
    
    # Convert to tensor
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    
    return tensor


def visualize_single_attention(model, src_tensor, src_vocab, trg_vocab, 
                               max_len=50, device='cpu', title='Attention Heatmap',
                               save_path=None):
    """
    Visualize attention weights for a single example
    
    Args:
        model: Trained model with attention mechanism
        src_tensor: Source input tensor [1, src_len]
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        max_len: Maximum generation length
        device: Torch device
        title: Plot title
        save_path: Path to save visualization
        
    Returns:
        Tuple of (generated_tokens, attention_matrix)
    """
    model.eval()
    
    with torch.no_grad():
        # Generate with attention
        try:
            predictions, attention_weights = model.generate(
                src_tensor, max_len, SOS_IDX, EOS_IDX
            )
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return [], None
        
        # Decode source tokens
        src_indices = src_tensor[0].cpu().numpy()
        src_tokens = decode_sequence(src_indices, src_vocab)
        
        # Decode generated tokens
        pred_indices = predictions[0].cpu().numpy()
        generated_tokens = decode_sequence(pred_indices, trg_vocab)
        
        # Stack attention weights into matrix
        # attention_weights is a list of tensors [1, src_len]
        if len(attention_weights) == 0:
            print("⚠️ No attention weights returned")
            return generated_tokens, None
        
        attention_matrix = []
        for att in attention_weights:
            if att.dim() == 2:
                # Shape: [batch=1, src_len]
                attention_matrix.append(att[0].cpu().numpy())
            elif att.dim() == 1:
                # Shape: [src_len]
                attention_matrix.append(att.cpu().numpy())
            else:
                print(f"⚠️ Unexpected attention shape: {att.shape}")
                continue
        
        if len(attention_matrix) == 0:
            print("⚠️ No valid attention weights")
            return generated_tokens, None
        
        # Convert to numpy array: [trg_len, src_len]
        attention_matrix = np.array(attention_matrix)
        
        # Trim to actual lengths (remove padding)
        if len(generated_tokens) > 0 and len(src_tokens) > 0:
            attention_matrix = attention_matrix[:len(generated_tokens), :len(src_tokens)]
        
        # Create visualization
        if save_path:
            fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.5), 
                                           max(8, len(generated_tokens) * 0.4)))
            
            # Create heatmap
            sns.heatmap(
                attention_matrix,
                xticklabels=src_tokens if len(src_tokens) <= 20 else False,
                yticklabels=generated_tokens if len(generated_tokens) <= 20 else False,
                cmap='YlOrRd',
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=1,
                linewidths=0.3,
                linecolor='lightgray'
            )
            
            ax.set_xlabel('Source Tokens (Docstring)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Generated Tokens (Code)', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Rotate labels if they exist
            if len(src_tokens) <= 20:
                plt.xticks(rotation=45, ha='right', fontsize=9)
            if len(generated_tokens) <= 20:
                plt.yticks(rotation=0, fontsize=9)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return generated_tokens, attention_matrix


def calculate_bleu_manual(prediction: List[str], reference: List[str], max_n: int = 4) -> float:
    """
    Calculate BLEU score manually (fallback if sacrebleu not available)
    
    Args:
        prediction: list of predicted tokens
        reference: list of reference tokens
        max_n: maximum n-gram order (default: 4)
    
    Returns:
        BLEU score (0-100)
    """
    from collections import Counter
    import math
    
    # Precision for each n-gram
    precisions = []
    
    for n in range(1, max_n + 1):
        # Get n-grams
        pred_ngrams = [tuple(prediction[i:i+n]) for i in range(len(prediction) - n + 1)]
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        
        if len(pred_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        matches = sum((pred_counter & ref_counter).values())
        total = len(pred_ngrams)
        
        precisions.append(matches / total if total > 0 else 0.0)
    
    # Brevity penalty
    pred_len = len(prediction)
    ref_len = len(reference)
    
    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0.0
    
    bleu = bp * geo_mean * 100
    return bleu


def check_python_syntax(code_string: str) -> Tuple[bool, str]:
    """
    Check if Python code is syntactically valid
    
    Args:
        code_string: Python code as string
    
    Returns:
        (is_valid, error_message)
    """
    try:
        ast.parse(code_string)
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def analyze_errors(predictions: List[str], references: List[str]) -> Dict:
    """
    Analyze common errors in predictions
    
    Args:
        predictions: list of predicted code strings
        references: list of reference code strings
    
    Returns:
        Dictionary with error statistics
    """
    stats = {
        'total': len(predictions),
        'syntax_errors': 0,
        'indentation_errors': 0,
        'length_mismatch': 0,
        'common_mistakes': []
    }
    
    for pred, ref in zip(predictions, references):
        # Check syntax
        is_valid, error = check_python_syntax(pred)
        if not is_valid:
            stats['syntax_errors'] += 1
            if 'indent' in error.lower():
                stats['indentation_errors'] += 1
        
        # Check length
        if abs(len(pred.split()) - len(ref.split())) > 5:
            stats['length_mismatch'] += 1
    
    stats['syntax_error_rate'] = stats['syntax_errors'] / stats['total'] * 100
    stats['indentation_error_rate'] = stats['indentation_errors'] / stats['total'] * 100
    
    return stats


def plot_training_curves(train_losses: List[float], val_losses: List[float], 
                         save_path: str, model_name: str = "Model"):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: list of training losses per epoch
        val_losses: list of validation losses per epoch
        save_path: path to save the plot
        model_name: name of the model for title
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    plt.plot(best_epoch, best_val_loss, 'r*', markersize=15, 
             label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_table(results: List[Dict]) -> str:
    """
    Create a formatted comparison table
    
    Args:
        results: list of result dictionaries from different models
    
    Returns:
        Formatted table string
    """
    table = "\n" + "="*90 + "\n"
    table += "MODEL COMPARISON - EVALUATION METRICS\n"
    table += "="*90 + "\n"
    table += f"{'Model':<20} {'Token Acc (%)':<15} {'BLEU':<12} {'Exact Match (%)':<18} {'Syntax Valid (%)':<18}\n"
    table += "-"*90 + "\n"
    
    for result in results:
        table += f"{result['model']:<20} "
        table += f"{result.get('token_accuracy', 0):>13.2f}  "
        table += f"{result.get('bleu_score', 0):>10.2f}  "
        table += f"{result.get('exact_match_accuracy', 0):>16.2f}  "
        table += f"{result.get('syntax_accuracy', 0):>16.2f}\n"
    
    table += "="*90 + "\n"
    
    return table


def save_metrics_json(metrics: Dict, filepath: str):
    """Save metrics to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


def load_metrics_json(filepath: str) -> Dict:
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_edit_distance(str1: str, str2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings
    
    Args:
        str1: first string
        str2: second string
    
    Returns:
        Edit distance (number of operations)
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def get_model_summary(model: torch.nn.Module) -> Dict:
    """
    Get summary of model parameters
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"