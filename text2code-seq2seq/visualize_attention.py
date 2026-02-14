"""
Attention Visualization Script
Visualizes attention weights for LSTM Attention Seq2Seq model

Requirements:
✅ Minimum 3 test examples
✅ Heatmap creation using matplotlib/seaborn
✅ Saves images as attention_example_1.png, attention_example_2.png, etc.

Usage:
    python visualize_attention.py --model_path checkpoints/best_model.pt
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.lstm_attention import create_lstm_attention_model
from data_preprocessing import (
    load_vocab, 
    tokenize, 
    PAD_IDX, 
    SOS_IDX, 
    EOS_IDX, 
    UNK_IDX
)
from utils import (
    decode_sequence,
    prepare_test_example,
    visualize_single_attention,
    indices_to_sentence
)


# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_trained_model(model_path, src_vocab, trg_vocab, device='cpu'):
    """
    Load a trained LSTM Attention model from checkpoint
    
    Args:
        model_path: Path to saved model checkpoint
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    # Create model architecture
    model = create_lstm_attention_model(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(trg_vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        device=device
    )
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"✓ Model loaded from: {model_path}")
    else:
        print(f"⚠ Model checkpoint not found at: {model_path}")
        print("  Using randomly initialized model for demonstration")
    
    model.eval()
    return model


def visualize_attention_for_examples(model, test_examples, src_vocab, trg_vocab, 
                                     output_dir='results/attention_viz', device='cpu'):
    """
    Generate attention visualizations for multiple test examples
    
    Args:
        model: Trained model with attention mechanism
        test_examples: List of test description strings
        src_vocab: Source vocabulary
        trg_vocab: Target vocabulary
        output_dir: Directory to save visualization images
        device: Torch device
        
    Returns:
        List of tuples: (input_text, generated_code, attention_matrix)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print("\n" + "="*80)
    print("ATTENTION VISUALIZATION")
    print("="*80 + "\n")
    
    for idx, example in enumerate(test_examples, 1):
        print(f"\n{'─'*80}")
        print(f"Example {idx}/{len(test_examples)}")
        print(f"{'─'*80}")
        print(f"Input: {example}")
        
        # Prepare input tensor
        src_tensor = prepare_test_example(example, src_vocab, device, max_len=50)
        
        # Generate and visualize
        save_path = os.path.join(output_dir, f"attention_example_{idx}.png")
        generated_tokens, attention_matrix = visualize_single_attention(
            model=model,
            src_tensor=src_tensor,
            src_vocab=src_vocab,
            trg_vocab=trg_vocab,
            max_len=50,
            device=device,
            title=f"Attention Heatmap - Example {idx}",
            save_path=save_path
        )
        
        # Print generated code
        generated_code = ' '.join(generated_tokens)
        print(f"Generated: {generated_code}")
        print(f"Saved visualization: {save_path}")
        
        results.append((example, generated_code, attention_matrix))
    
    print("\n" + "="*80)
    print(f"✓ All {len(test_examples)} visualizations saved to: {output_dir}/")
    print("="*80 + "\n")
    
    return results


def create_comprehensive_test_examples():
    """
    Create diverse test examples covering different programming patterns
    
    Returns:
        List of test description strings
    """
    examples = [
        # Example 1: Simple arithmetic function
        "create a function to add two numbers",
        
        # Example 2: List operation
        "sort a list of numbers in ascending order",
        
        # Example 3: Conditional logic
        "check if a number is even or odd",
        
        # Example 4: String manipulation
        "convert a string to uppercase",
        
        # Example 5: Loop/iteration
        "calculate the sum of numbers in a list",
        
        # Example 6: Mathematical function
        "calculate the square of a number",
        
        # Example 7: Boolean function
        "check if a string is empty"
    ]
    
    return examples


def plot_attention_comparison(results, output_path='results/attention_viz/comparison.png'):
    """
    Create a side-by-side comparison of attention patterns
    
    Args:
        results: List of (input, output, attention_matrix) tuples
        output_path: Path to save comparison figure
    """
    num_examples = min(3, len(results))  # Show max 3 examples
    
    fig, axes = plt.subplots(1, num_examples, figsize=(18, 6))
    
    if num_examples == 1:
        axes = [axes]
    
    for idx, (input_text, output_text, attention_matrix) in enumerate(results[:num_examples]):
        if attention_matrix is None:
            continue
            
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            ax=axes[idx],
            cmap='YlGnBu',
            cbar=True,
            square=False,
            vmin=0,
            vmax=1
        )
        
        # Truncate title if too long
        title = input_text if len(input_text) < 30 else input_text[:27] + "..."
        axes[idx].set_title(f"Ex{idx+1}: {title}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Source', fontsize=9)
        axes[idx].set_ylabel('Target', fontsize=9)
        axes[idx].tick_params(labelsize=7)
    
    plt.suptitle('Attention Pattern Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_path}")
    plt.close()


def analyze_attention_patterns(results):
    """
    Analyze and print statistics about attention patterns
    
    Args:
        results: List of (input, output, attention_matrix) tuples
    """
    print("\n" + "="*80)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*80 + "\n")
    
    for idx, (input_text, output_text, attention_matrix) in enumerate(results, 1):
        if attention_matrix is None:
            continue
        
        print(f"\nExample {idx}:")
        print(f"Input length: {attention_matrix.shape[1]} tokens")
        print(f"Output length: {attention_matrix.shape[0]} tokens")
        
        # Calculate attention statistics
        max_attention = np.max(attention_matrix)
        mean_attention = np.mean(attention_matrix)
        std_attention = np.std(attention_matrix)
        
        print(f"Max attention weight: {max_attention:.4f}")
        print(f"Mean attention weight: {mean_attention:.4f}")
        print(f"Std attention weight: {std_attention:.4f}")
        
        # Find most attended source tokens
        sum_attention = attention_matrix.sum(axis=0)
        top_k = 3
        top_indices = np.argsort(sum_attention)[-top_k:][::-1]
        
        print(f"Most attended source positions: {top_indices.tolist()}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to run attention visualization"""
    parser = argparse.ArgumentParser(description='Visualize attention weights for Seq2Seq model')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--src_vocab', type=str, default='checkpoints/src_vocab.pkl',
                       help='Path to source vocabulary')
    parser.add_argument('--trg_vocab', type=str, default='checkpoints/trg_vocab.pkl',
                       help='Path to target vocabulary')
    parser.add_argument('--output_dir', type=str, default='results/attention_viz',
                       help='Directory to save attention visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='Number of test examples to visualize')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    if os.path.exists(args.src_vocab) and os.path.exists(args.trg_vocab):
        src_vocab = load_vocab(args.src_vocab)
        trg_vocab = load_vocab(args.trg_vocab)
        print(f"✓ Source vocab size: {len(src_vocab)}")
        print(f"✓ Target vocab size: {len(trg_vocab)}")
    else:
        print("⚠ Vocabulary files not found. Please train the model first.")
        print("  Creating dummy vocabularies for demonstration...")
        from data_preprocessing import Vocabulary
        src_vocab = Vocabulary()
        trg_vocab = Vocabulary()
        # Add some basic tokens
        for word in "create function add two numbers sort list check if string".split():
            src_vocab.add_sentence(word)
        for word in "def add a b return + ( ) : sorted nums".split():
            trg_vocab.add_sentence(word)
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = load_trained_model(args.model_path, src_vocab, trg_vocab, device)
    
    # Create test examples
    print("\nPreparing test examples...")
    test_examples = create_comprehensive_test_examples()[:args.num_examples]
    print(f"✓ Created {len(test_examples)} test examples")
    
    # Visualize attention
    results = visualize_attention_for_examples(
        model=model,
        test_examples=test_examples,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        output_dir=args.output_dir,
        device=device
    )
    
    # Create comparison visualization
    if len(results) >= 2:
        comparison_path = os.path.join(args.output_dir, 'comparison.png')
        plot_attention_comparison(results, comparison_path)
    
    # Analyze patterns
    analyze_attention_patterns(results)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print(f"Generated files:")
    for i in range(1, len(test_examples) + 1):
        print(f"  - attention_example_{i}.png")
    if len(results) >= 2:
        print(f"  - comparison.png")
    print("\n")


if __name__ == "__main__":
    main()