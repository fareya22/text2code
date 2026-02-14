"""
Comprehensive Evaluation Script for Seq2Seq Models
- Token-level Accuracy
- BLEU Score
- Exact Match Accuracy
- Syntax Validation (AST)
- Performance vs Docstring Length
"""

import torch
import json
import os
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sacrebleu.metrics import BLEU

from data_preprocessing import (
    load_and_preprocess_data,
    CodeDataset,
    load_vocab,
    collate_batch,
    indices_to_sentence,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX
)
from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model
from models.transformer import create_transformer_model


class Evaluator:
    def __init__(self, model, model_name, device, src_vocab, trg_vocab):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        
    def decode_sequence(self, indices):
        """Convert indices to tokens, removing special tokens"""
        tokens = []
        for idx in indices:
            if idx == EOS_IDX or idx == PAD_IDX:
                break
            if idx == SOS_IDX:
                continue
            token = self.trg_vocab.itos.get(idx, '<UNK>')
            tokens.append(token)
        return tokens
    
    def calculate_token_accuracy(self, predicted, target):
        """Calculate token-level accuracy"""
        correct = 0
        total = 0
        
        for pred_seq, tgt_seq in zip(predicted, target):
            # Skip SOS token
            pred_tokens = pred_seq[1:]
            tgt_tokens = tgt_seq[1:]
            
            # Compare up to min length
            min_len = min(len(pred_tokens), len(tgt_tokens))
            for i in range(min_len):
                if tgt_tokens[i] == EOS_IDX or tgt_tokens[i] == PAD_IDX:
                    break
                total += 1
                if pred_tokens[i] == tgt_tokens[i]:
                    correct += 1
                    
        return (correct / total * 100) if total > 0 else 0.0
    
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score using sacrebleu"""
        bleu = BLEU()
        
        pred_texts = []
        ref_texts = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.decode_sequence(pred)
            ref_tokens = self.decode_sequence(ref)
            
            pred_text = ' '.join(pred_tokens)
            ref_text = ' '.join(ref_tokens)
            
            pred_texts.append(pred_text)
            ref_texts.append([ref_text])  # sacrebleu expects list of references
        
        score = bleu.corpus_score(pred_texts, ref_texts)
        return score.score
    
    def calculate_exact_match(self, predictions, references):
        """Calculate exact match accuracy"""
        matches = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.decode_sequence(pred)
            ref_tokens = self.decode_sequence(ref)
            
            if pred_tokens == ref_tokens:
                matches += 1
                
        return (matches / total * 100) if total > 0 else 0.0
    
    def check_syntax_valid(self, code_tokens):
        """Check if generated code is syntactically valid using AST"""
        try:
            code_str = ' '.join(code_tokens)
            # Replace special tokens
            code_str = code_str.replace('NEWLINE', '\n').replace('INDENT', '\t')
            ast.parse(code_str)
            return True
        except:
            return False
    
    def calculate_syntax_accuracy(self, predictions):
        """Calculate percentage of syntactically valid code"""
        valid = 0
        total = len(predictions)
        
        for pred in predictions:
            pred_tokens = self.decode_sequence(pred)
            if self.check_syntax_valid(pred_tokens):
                valid += 1
                
        return (valid / total * 100) if total > 0 else 0.0
    
    def evaluate_by_length(self, test_loader, max_len=150):
        """Evaluate performance vs docstring length"""
        self.model.eval()
        
        length_buckets = defaultdict(lambda: {'predictions': [], 'references': []})
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {self.model_name} by length"):
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)
                
                # Generate predictions
                if 'attention' in self.model_name.lower():
                    predictions, _ = self.model.generate(src, max_len, SOS_IDX, EOS_IDX)
                else:
                    predictions = self.model.generate(src, max_len, SOS_IDX, EOS_IDX)
                
                # Group by source length
                src_lengths = (src != PAD_IDX).sum(dim=1).cpu().numpy()
                
                for i in range(len(src)):
                    src_len = src_lengths[i]
                    bucket = (src_len // 10) * 10  # Bucket by 10s
                    
                    length_buckets[bucket]['predictions'].append(predictions[i].cpu().numpy())
                    length_buckets[bucket]['references'].append(trg[i].cpu().numpy())
        
        # Calculate BLEU for each bucket
        results = {}
        for length, data in sorted(length_buckets.items()):
            if len(data['predictions']) >= 5:  # Only buckets with enough samples
                bleu_score = self.calculate_bleu(data['predictions'], data['references'])
                results[length] = {
                    'bleu': bleu_score,
                    'count': len(data['predictions'])
                }
        
        return results
    
    def evaluate(self, test_loader, max_len=150):
        """Complete evaluation on test set"""
        self.model.eval()
        
        all_predictions = []
        all_references = []
        
        print(f"\nEvaluating {self.model_name}...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)
                
                # Generate predictions
                if 'attention' in self.model_name.lower():
                    predictions, _ = self.model.generate(src, max_len, SOS_IDX, EOS_IDX)
                else:
                    predictions = self.model.generate(src, max_len, SOS_IDX, EOS_IDX)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_references.extend(trg.cpu().numpy())
        
        # Calculate metrics
        print("Calculating metrics...")
        token_acc = self.calculate_token_accuracy(all_predictions, all_references)
        bleu_score = self.calculate_bleu(all_predictions, all_references)
        exact_match = self.calculate_exact_match(all_predictions, all_references)
        syntax_acc = self.calculate_syntax_accuracy(all_predictions)
        
        results = {
            'model': self.model_name,
            'token_accuracy': token_acc,
            'bleu_score': bleu_score,
            'exact_match_accuracy': exact_match,
            'syntax_accuracy': syntax_acc,
            'num_samples': len(all_predictions)
        }
        
        return results, all_predictions, all_references


def plot_performance_vs_length(results_by_model, save_path):
    """Plot performance vs docstring length for all models"""
    plt.figure(figsize=(12, 6))
    
    for model_name, length_results in results_by_model.items():
        lengths = sorted(length_results.keys())
        bleu_scores = [length_results[l]['bleu'] for l in lengths]
        
        plt.plot(lengths, bleu_scores, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Docstring Length (tokens)', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('Performance vs Docstring Length', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance vs length plot saved to {save_path}")


def print_comparison_table(all_results):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON - EVALUATION RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'Token Acc':<12} {'BLEU':<12} {'Exact Match':<12} {'Syntax Valid':<12}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['model']:<20} "
              f"{result['token_accuracy']:>10.2f}% "
              f"{result['bleu_score']:>10.2f}  "
              f"{result['exact_match_accuracy']:>10.2f}% "
              f"{result['syntax_accuracy']:>10.2f}%")
    
    print("="*80 + "\n")


def save_sample_outputs(predictions, references, src_vocab, trg_vocab, save_path, num_samples=10):
    """Save sample predictions vs references"""
    with open(save_path, 'w') as f:
        f.write("SAMPLE PREDICTIONS\n")
        f.write("="*80 + "\n\n")
        
        for i in range(min(num_samples, len(predictions))):
            pred_tokens = []
            ref_tokens = []
            
            # Decode prediction
            for idx in predictions[i]:
                if idx == EOS_IDX or idx == PAD_IDX:
                    break
                if idx != SOS_IDX:
                    pred_tokens.append(trg_vocab.itos.get(idx, '<UNK>'))
            
            # Decode reference
            for idx in references[i]:
                if idx == EOS_IDX or idx == PAD_IDX:
                    break
                if idx != SOS_IDX:
                    ref_tokens.append(trg_vocab.itos.get(idx, '<UNK>'))
            
            f.write(f"Example {i+1}:\n")
            f.write(f"PREDICTED: {' '.join(pred_tokens)}\n")
            f.write(f"REFERENCE: {' '.join(ref_tokens)}\n")
            f.write(f"MATCH: {'✓' if pred_tokens == ref_tokens else '✗'}\n")
            f.write("-"*80 + "\n\n")


def main():
    # Configuration
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    config = {
        "num_test": 100,
        "max_docstring_len": 100,
        "max_code_len": 150,
        "batch_size": 15,
        "embedding_dim": 256,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.5,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    print("Loading vocabularies...")
    docstring_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, 'docstring_vocab.pkl'))
    code_vocab = load_vocab(os.path.join(CHECKPOINT_DIR, 'code_vocab.pkl'))
    
    # Load test data
    print("Loading test data...")
    _, _, test_data, _, _ = load_and_preprocess_data(
        num_train=100,  # Don't need train/val for evaluation
        num_val=100,
        num_test=config['num_test'],
        max_docstring_len=config['max_docstring_len'],
        max_code_len=config['max_code_len']
    )
    
    test_dataset = CodeDataset(test_data, docstring_vocab, code_vocab,
                               config['max_docstring_len'], config['max_code_len'])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=collate_batch, num_workers=0
    )
    
    # Models to evaluate
    models_config = [
        ('vanilla_rnn', create_vanilla_rnn_model, {'num_layers': None}),
        ('lstm', create_lstm_model, {'num_layers': config['num_layers']}),
        ('lstm_attention', create_lstm_attention_model, {'num_layers': config['num_layers']}),
        ('transformer', create_transformer_model, {'num_layers': config['num_layers']})  # Bonus
    ]
    
    all_results = []
    results_by_length = {}
    
    for model_name, model_factory, extra_config in models_config:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*60}")
        
        # Create model
        if extra_config.get('num_layers') is None:
            model = model_factory(
                input_vocab_size=len(docstring_vocab),
                output_vocab_size=len(code_vocab),
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                device=device
            )
        else:
            model = model_factory(
                input_vocab_size=len(docstring_vocab),
                output_vocab_size=len(code_vocab),
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                device=device
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate
        evaluator = Evaluator(model, model_name, device, docstring_vocab, code_vocab)
        results, predictions, references = evaluator.evaluate(
            test_loader, max_len=config['max_code_len']
        )
        all_results.append(results)
        
        # Evaluate by length
        print("Evaluating performance vs docstring length...")
        length_results = evaluator.evaluate_by_length(
            test_loader, max_len=config['max_code_len']
        )
        results_by_length[model_name] = length_results
        
        # Save sample outputs
        sample_path = os.path.join(RESULTS_DIR, f'{model_name}_samples.txt')
        save_sample_outputs(predictions, references, docstring_vocab, code_vocab, sample_path)
        print(f"Sample outputs saved to {sample_path}")
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Save results to JSON
    results_json_path = os.path.join(RESULTS_DIR, 'metrics.json')
    with open(results_json_path, 'w') as f:
        json.dump({
            'overall_results': all_results,
            'performance_by_length': {k: {int(kk): vv for kk, vv in v.items()} 
                                      for k, v in results_by_length.items()}
        }, f, indent=2)
    print(f"Results saved to {results_json_path}")
    
    # Plot performance vs length
    if results_by_length:
        plot_path = os.path.join(RESULTS_DIR, 'performance_vs_length.png')
        plot_performance_vs_length(results_by_length, plot_path)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
