"""
Attention Visualization
Generate heatmaps showing alignment between docstring and code
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from data_preprocessing import (
    load_vocab,
    CodeDataset,
    indices_to_sentence,
    load_and_preprocess_data
)
from models.lstm_attention import create_lstm_attention_model


def visualize_attention(input_tokens, output_tokens, attention_weights, 
                       save_path, title="Attention Visualization"):
    """
    Create attention heatmap
    
    Args:
        input_tokens: List of input tokens (docstring)
        output_tokens: List of output tokens (generated code)
        attention_weights: Tensor of shape (output_len, input_len)
        save_path: Path to save the plot
        title: Title for the plot
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap='YlOrRd',
        cbar=True,
        ax=ax,
        square=False,
        linewidths=0.5
    )
    
    # Formatting
    ax.set_xlabel('Input (Docstring)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Output (Generated Code)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualization saved to {save_path}")


def analyze_attention_patterns(input_tokens, output_tokens, attention_weights):
    """
    Analyze attention patterns and provide insights
    
    Args:
        input_tokens: List of input tokens
        output_tokens: List of output tokens
        attention_weights: Attention weight matrix
    
    Returns:
        Analysis results as a dictionary
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    analysis = {
        'max_attention_pairs': [],
        'attention_statistics': {},
        'insights': []
    }
    
    # Find max attention for each output token
    for i, out_token in enumerate(output_tokens):
        if i >= len(attention_weights):
            break
        max_idx = np.argmax(attention_weights[i])
        max_weight = attention_weights[i][max_idx]
        
        if max_idx < len(input_tokens):
            analysis['max_attention_pairs'].append({
                'output_token': out_token,
                'input_token': input_tokens[max_idx],
                'attention_weight': float(max_weight)
            })
    
    # Calculate statistics
    analysis['attention_statistics'] = {
        'mean': float(np.mean(attention_weights)),
        'std': float(np.std(attention_weights)),
        'max': float(np.max(attention_weights)),
        'min': float(np.min(attention_weights))
    }
    
    # Generate insights
    # Check if attention is focused or distributed
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    avg_entropy = np.mean(entropy)
    
    if avg_entropy < 1.0:
        analysis['insights'].append("Attention is highly focused (low entropy)")
    elif avg_entropy > 2.5:
        analysis['insights'].append("Attention is distributed (high entropy)")
    else:
        analysis['insights'].append("Attention shows moderate focus")
    
    # Check for diagonal patterns (sequential alignment)
    diagonal_strength = 0
    for i in range(min(len(output_tokens), len(input_tokens))):
        if i < len(attention_weights):
            diagonal_strength += attention_weights[i][i]
    diagonal_strength /= min(len(output_tokens), len(input_tokens))
    
    if diagonal_strength > 0.3:
        analysis['insights'].append("Strong sequential alignment detected")
    
    return analysis


def visualize_multiple_examples(model, test_loader, docstring_vocab, code_vocab, 
                                device, num_examples=3, save_dir='attention_plots'):
    """
    Generate attention visualizations for multiple test examples
    
    Args:
        model: Trained attention model
        test_loader: Test data loader
        docstring_vocab: Docstring vocabulary
        code_vocab: Code vocabulary
        device: Device to run on
        num_examples: Number of examples to visualize
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    sos_idx = code_vocab.word2idx['<SOS>']
    eos_idx = code_vocab.word2idx['<EOS>']
    
    examples_visualized = 0
    all_analyses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if examples_visualized >= num_examples:
                break
            
            src = batch['docstring'].to(device)
            trg = batch['code'].to(device)
            
            # Generate with attention
            generated, attentions = model.generate(src, max_len=82, 
                                                  sos_idx=sos_idx, 
                                                  eos_idx=eos_idx)
            
            # Process first example in batch
            for i in range(min(src.shape[0], num_examples - examples_visualized)):
                # Get tokens
                input_indices = src[i].cpu().numpy()
                output_indices = generated[i].cpu().numpy()
                reference_indices = trg[i].cpu().numpy()
                
                # Convert to words
                input_words = []
                for idx in input_indices:
                    if idx == 0:  # PAD
                        break
                    word = docstring_vocab.idx2word.get(idx, '<UNK>')
                    input_words.append(word)
                
                output_words = []
                for idx in output_indices:
                    if idx == eos_idx or idx == 0:
                        break
                    if idx != sos_idx:
                        word = code_vocab.idx2word.get(idx, '<UNK>')
                        output_words.append(word)
                
                reference_words = []
                for idx in reference_indices:
                    if idx == eos_idx or idx == 0:
                        break
                    if idx != sos_idx:
                        word = code_vocab.idx2word.get(idx, '<UNK>')
                        reference_words.append(word)
                
                # Stack attention weights
                # attentions is a list of (batch_size, src_len) tensors
                attention_matrix = torch.stack([att[i] for att in attentions[:len(output_words)]])
                # attention_matrix: (output_len, src_len)
                
                # Trim to actual lengths
                attention_matrix = attention_matrix[:len(output_words), :len(input_words)]
                
                # Visualize
                title = f"Attention Visualization - Example {examples_visualized + 1}"
                save_path = os.path.join(save_dir, f'attention_example_{examples_visualized + 1}.png')
                
                visualize_attention(
                    input_words,
                    output_words,
                    attention_matrix,
                    save_path,
                    title
                )
                
                # Analyze
                analysis = analyze_attention_patterns(input_words, output_words, attention_matrix)
                analysis['example_id'] = examples_visualized + 1
                analysis['input_text'] = ' '.join(input_words)
                analysis['generated_text'] = ' '.join(output_words)
                analysis['reference_text'] = ' '.join(reference_words)
                
                all_analyses.append(analysis)
                
                # Print analysis
                print(f"\n{'='*60}")
                print(f"Example {examples_visualized + 1} Analysis")
                print(f"{'='*60}")
                print(f"\nInput: {analysis['input_text']}")
                print(f"Generated: {analysis['generated_text']}")
                print(f"Reference: {analysis['reference_text']}")
                print(f"\nInsights:")
                for insight in analysis['insights']:
                    print(f"  - {insight}")
                print(f"\nTop 5 Attention Alignments:")
                for j, pair in enumerate(analysis['max_attention_pairs'][:5]):
                    print(f"  {j+1}. '{pair['output_token']}' <- '{pair['input_token']}' "
                          f"(weight: {pair['attention_weight']:.3f})")
                
                examples_visualized += 1
                
                if examples_visualized >= num_examples:
                    break
    
    return all_analyses


def main():
    """Main visualization function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    print("\nLoading vocabularies...")
    docstring_vocab = load_vocab('checkpoints/docstring_vocab.pkl')
    code_vocab = load_vocab('checkpoints/code_vocab.pkl')
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_data, _, _ = load_and_preprocess_data(
        num_train=100,
        num_val=100,
        num_test=1000,
        max_docstring_len=50,
        max_code_len=80
    )
    
    test_dataset = CodeDataset(test_data, docstring_vocab, code_vocab, 50, 80)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Load attention model
    print("\nLoading LSTM + Attention model...")
    checkpoint_path = 'checkpoints/lstm_attention_best.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    model = create_lstm_attention_model(
        input_vocab_size=len(docstring_vocab),
        output_vocab_size=len(code_vocab),
        embedding_dim=256,
        hidden_dim=256,
        num_layers=1,
        device=device
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Attention Visualizations")
    print("="*60)
    
    analyses = visualize_multiple_examples(
        model, 
        test_loader, 
        docstring_vocab, 
        code_vocab, 
        device,
        num_examples=5,
        save_dir='attention_plots'
    )
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nAttention heatmaps saved in: attention_plots/")
    print("  - attention_example_1.png")
    print("  - attention_example_2.png")
    print("  - ...")


if __name__ == "__main__":
    main()
