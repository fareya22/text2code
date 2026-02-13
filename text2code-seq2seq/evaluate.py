"""
Evaluation Script
Calculate BLEU score, token accuracy, exact match accuracy, AST validation,
and comprehensive error analysis
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import json
import os
import ast
from collections import Counter

from data_preprocessing import (
    load_vocab,
    CodeDataset,
    indices_to_sentence,
    load_and_preprocess_data
)
from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model
from models.transformer import create_transformer_model
from models.transformer import create_transformer_model


# ============ COMPREHENSIVE EVALUATION FUNCTIONS ============

def validate_syntax_ast(generated_tokens):
    """
    Check if generated tokens form valid Python syntax using ast.parse().
    
    Args:
        generated_tokens: list of string tokens or space-separated string
    
    Returns:
        True if valid Python syntax, False otherwise
    """
    if isinstance(generated_tokens, list):
        code = " ".join(generated_tokens)
    else:
        code = generated_tokens
    
    code = code.replace("NEWLINE", "\n").replace("INDENT", "    ")
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def compute_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    """
    Compute BLEU score between reference and hypothesis token lists.
    Uses smoothed BLEU with brevity penalty.
    
    Args:
        reference_tokens: list of reference tokens
        hypothesis_tokens: list of generated tokens
        max_n: max n-gram to compute
    
    Returns:
        BLEU score (0.0 to 1.0)
    """
    if len(hypothesis_tokens) == 0:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        hyp_ngrams = Counter()

        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i:i + n])
            ref_ngrams[ngram] += 1

        for i in range(len(hypothesis_tokens) - n + 1):
            ngram = tuple(hypothesis_tokens[i:i + n])
            hyp_ngrams[ngram] += 1

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            # Add-1 smoothing for n > 1
            if n > 1 and clipped == 0:
                precisions.append(1.0 / (total + 1))
            else:
                precisions.append(clipped / total)

    # Geometric mean of precisions
    if min(precisions) == 0:
        return 0.0

    log_avg = sum(np.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    if hyp_len < ref_len:
        bp = np.exp(1 - ref_len / hyp_len)

    bleu = bp * np.exp(log_avg)
    return bleu


def compute_exact_match(reference_tokens, hypothesis_tokens):
    """Check if generated code exactly matches reference."""
    if isinstance(reference_tokens, list) and isinstance(hypothesis_tokens, list):
        return reference_tokens == hypothesis_tokens
    else:
        return str(reference_tokens).strip() == str(hypothesis_tokens).strip()


def compute_token_accuracy(prediction, reference):
    """
    Compute token-level accuracy.
    
    Args:
        prediction: predicted token string or list
        reference: reference token string or list
    
    Returns:
        Accuracy as fraction (0.0 to 1.0)
    """
    if isinstance(prediction, str):
        pred_tokens = prediction.split()
    else:
        pred_tokens = prediction
    
    if isinstance(reference, str):
        ref_tokens = reference.split()
    else:
        ref_tokens = reference
    
    min_len = min(len(pred_tokens), len(ref_tokens))
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for p, r in zip(pred_tokens[:min_len], ref_tokens[:min_len]) if p == r)
    return matches / max(len(pred_tokens), len(ref_tokens))


def analyze_error_types(predictions, references):
    """
    Classify common error types: syntax, indentation, operators, missing/extra tokens.
    
    Returns:
        Dictionary with error category counts
    """
    errors = {
        'syntax_errors': 0,
        'missing_indentation': 0,
        'incorrect_operators': 0,
        'missing_tokens': 0,
        'extra_tokens': 0,
        'total_examples': 0,
        'examples': []
    }

    for pred, ref in zip(predictions, references):
        errors['total_examples'] += 1
        
        if pred.strip() == ref.strip():
            continue

        error_info = {
            'predicted': pred[:100],
            'reference': ref[:100],
            'error_types': []
        }

        # Check for missing indentation
        if "INDENT" in ref and "INDENT" not in pred:
            errors['missing_indentation'] += 1
            error_info['error_types'].append('missing_indentation')

        # Check for incorrect operators
        operators = ["+", "-", "*", "/", "==", "!=", ">=", "<=", ">", "<",
                     "and", "or", "not", "in", "is"]
        ref_ops = set(t for t in ref.split() if t in operators)
        pred_ops = set(t for t in pred.split() if t in operators)
        if ref_ops != pred_ops:
            errors['incorrect_operators'] += 1
            error_info['error_types'].append('incorrect_operators')

        # Check for length mismatch
        ref_len = len(ref.split())
        pred_len = len(pred.split())
        if pred_len < ref_len:
            errors['missing_tokens'] += 1
            error_info['error_types'].append('missing_tokens')
        elif pred_len > ref_len:
            errors['extra_tokens'] += 1
            error_info['error_types'].append('extra_tokens')

        # Check syntax
        if not validate_syntax_ast(pred.split()):
            errors['syntax_errors'] += 1
            error_info['error_types'].append('syntax_error')
        
        if len(errors['examples']) < 10:
            errors['examples'].append(error_info)

    return errors


def bleu_vs_docstring_length(predictions_list, references_list, docstring_lengths):
    """
    Compute BLEU score binned by docstring length.
    
    Args:
        predictions_list: list of predictions
        references_list: list of references
        docstring_lengths: list of docstring token counts
    
    Returns:
        Dictionary mapping length_bin -> avg_bleu
    """
    length_bleu = {}

    for pred, ref, src_len in zip(predictions_list, references_list, docstring_lengths):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        bleu = compute_bleu(ref_tokens, pred_tokens)
        
        # Bin by length (groups of 10)
        bin_key = (src_len // 10) * 10
        if bin_key not in length_bleu:
            length_bleu[bin_key] = []
        length_bleu[bin_key].append(bleu)

    # Average
    avg_length_bleu = {
        k: np.mean(v) for k, v in sorted(length_bleu.items())
    }
    return avg_length_bleu


class Evaluator:
    """Evaluation manager for Seq2Seq models"""
    
    def __init__(self, model, model_name, device, docstring_vocab, code_vocab):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.docstring_vocab = docstring_vocab
        self.code_vocab = code_vocab
        self.bleu = BLEU()
        
    def generate_predictions(self, dataloader, max_len=82):
        """Generate predictions for entire dataset"""
        self.model.eval()
        
        all_predictions = []
        all_references = []
        all_inputs = []
        
        sos_idx = self.code_vocab.word2idx['<SOS>']
        eos_idx = self.code_vocab.word2idx['<EOS>']
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'Generating with {self.model_name}'):
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)
                
                # Generate
                if 'attention' in self.model_name.lower():
                    generated, _ = self.model.generate(src, max_len, sos_idx, eos_idx)
                else:
                    generated = self.model.generate(src, max_len, sos_idx, eos_idx)
                
                # Convert to text
                for i in range(src.shape[0]):
                    # Input docstring
                    input_text = indices_to_sentence(
                        src[i].cpu().numpy(), 
                        self.docstring_vocab
                    )
                    
                    # Predicted code
                    pred_text = indices_to_sentence(
                        generated[i].cpu().numpy(), 
                        self.code_vocab
                    )
                    
                    # Reference code
                    ref_text = indices_to_sentence(
                        trg[i].cpu().numpy(),
                        self.code_vocab
                    )
                    
                    all_inputs.append(input_text)
                    all_predictions.append(pred_text)
                    all_references.append(ref_text)
        
        return all_inputs, all_predictions, all_references
    
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score"""
        # BLEU expects list of references for each prediction
        refs = [[ref] for ref in references]
        score = self.bleu.corpus_score(predictions, refs)
        return score.score
    
    def calculate_token_accuracy(self, predictions, references):
        """Calculate token-level accuracy"""
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Match tokens at each position
            for i in range(min(len(pred_tokens), len(ref_tokens))):
                total_tokens += 1
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
            
            # Count missing/extra tokens as wrong
            total_tokens += abs(len(pred_tokens) - len(ref_tokens))
        
        if total_tokens == 0:
            return 0.0
        
        return (correct_tokens / total_tokens) * 100
    
    def calculate_exact_match(self, predictions, references):
        """Calculate exact match accuracy"""
        exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                           if pred.strip() == ref.strip())
        return (exact_matches / len(predictions)) * 100
    
    def analyze_errors(self, inputs, predictions, references, num_samples=10):
        """Analyze types of errors"""
        errors = {
            'syntax_errors': [],
            'length_mismatches': [],
            'semantic_errors': []
        }
        
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            if pred.strip() != ref.strip():
                error_example = {
                    'index': i,
                    'input': inp,
                    'predicted': pred,
                    'reference': ref
                }
                
                # Length mismatch
                if abs(len(pred.split()) - len(ref.split())) > 5:
                    errors['length_mismatches'].append(error_example)
                
                # Syntax errors (basic check)
                if 'def' not in pred or ':' not in pred:
                    errors['syntax_errors'].append(error_example)
                else:
                    errors['semantic_errors'].append(error_example)
        
        return errors
    
    def evaluate(self, dataloader):
        """Complete comprehensive evaluation"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE EVALUATION: {self.model_name}")
        print(f"{'='*70}\n")
        
        # Generate predictions
        inputs, predictions, references = self.generate_predictions(dataloader)
        
        # ========== CORE METRICS ==========
        print("Computing core metrics...")
        
        # 1. BLEU Score (token-level n-gram quality)
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            bleu = compute_bleu(ref.split(), pred.split())
            bleu_scores.append(bleu)
        avg_bleu = np.mean(bleu_scores)
        
        # 2. Exact Match
        exact_matches = [compute_exact_match(p, r) for p, r in zip(predictions, references)]
        exact_match_rate = np.mean(exact_matches) * 100
        
        # 3. Token Accuracy (Fraction of correct tokens)
        token_accuracies = []
        for pred, ref in zip(predictions, references):
            acc = compute_token_accuracy(pred, ref)
            token_accuracies.append(acc)
        avg_token_accuracy = np.mean(token_accuracies) * 100
        
        # ========== SYNTAX VALIDATION ==========
        print("Validating syntax...")
        
        # 4. AST Valid Rate (Fraction of syntax-valid codes)
        ast_valid_count = 0
        for pred in predictions:
            if validate_syntax_ast(pred.split()):
                ast_valid_count += 1
        ast_valid_rate = (ast_valid_count / len(predictions)) * 100
        
        # ========== ERROR ANALYSIS ==========
        print("Analyzing errors...")
        
        # 5. Error Analysis (Classify common errors)
        error_analysis = analyze_error_types(predictions, references)
        
        # ========== LENGTH-BASED ANALYSIS ==========
        print("Computing length-based metrics...")
        
        # 6. BLEU vs Docstring Length
        docstring_lengths = [len(inp.split()) for inp in inputs]
        bleu_by_length = bleu_vs_docstring_length(predictions, references, docstring_lengths)
        
        # ========== ATTENTION WEIGHTS (if applicable) ==========
        attention_data = None
        if 'attention' in self.model_name.lower():
            print("Extracting attention weights...")
            attention_data = "Attention visualization available for attention-based model"
        
        # ========== COMPILE RESULTS ==========
        results = {
            'model_name': self.model_name,
            'total_examples': len(predictions),
            
            # Core Metrics
            'bleu_score': {
                'average': float(avg_bleu),
                'all_scores': [float(s) for s in bleu_scores[:100]]  # Save first 100
            },
            'exact_match_rate': float(exact_match_rate),
            'token_accuracy': float(avg_token_accuracy),
            
            # Syntax Validation
            'ast_valid_rate': float(ast_valid_rate),
            'ast_valid_count': int(ast_valid_count),
            
            # Error Analysis
            'error_analysis': {
                'total_examples': error_analysis['total_examples'],
                'syntax_errors': error_analysis['syntax_errors'],
                'missing_indentation': error_analysis['missing_indentation'],
                'incorrect_operators': error_analysis['incorrect_operators'],
                'missing_tokens': error_analysis['missing_tokens'],
                'extra_tokens': error_analysis['extra_tokens'],
                'error_examples': error_analysis['examples']
            },
            
            # Length-based Analysis
            'bleu_by_docstring_length': {
                str(k): float(v) for k, v in bleu_by_length.items()
            },
            
            # Attention info
            'has_attention': 'attention' in self.model_name.lower()
        }
        
        # ========== PRINT RESULTS ==========
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}\n")
        
        print("📊 CORE METRICS:")
        print(f"  BLEU Score (n-gram quality):        {avg_bleu:.4f}")
        print(f"  Token Accuracy (correct tokens):    {avg_token_accuracy:.2f}%")
        print(f"  Exact Match (perfect predictions):  {exact_match_rate:.2f}%")
        
        print(f"\n💻 SYNTAX VALIDATION:")
        print(f"  AST Valid Rate (valid syntax):      {ast_valid_rate:.2f}% ({ast_valid_count}/{len(predictions)})")
        
        print(f"\n⚠️  ERROR ANALYSIS:")
        print(f"  Total errors:                       {error_analysis['total_examples'] - sum(exact_matches)}")
        print(f"  Syntax errors:                      {error_analysis['syntax_errors']}")
        print(f"  Missing indentation:                {error_analysis['missing_indentation']}")
        print(f"  Incorrect operators:                {error_analysis['incorrect_operators']}")
        print(f"  Missing tokens:                     {error_analysis['missing_tokens']}")
        print(f"  Extra tokens:                       {error_analysis['extra_tokens']}")
        
        print(f"\n📏 BLEU vs DOCSTRING LENGTH:")
        for length_bin, bleu in sorted(bleu_by_length.items()):
            print(f"  {length_bin}-{length_bin+9} tokens:  {bleu:.4f}")
        
        print(f"\n🔍 SAMPLE PREDICTIONS (first 5):")
        for i in range(min(5, len(inputs))):
            match_status = "✓" if compute_exact_match(predictions[i], references[i]) else "✗"
            print(f"\n  Example {i+1}: {match_status}")
            print(f"    Input:      {inputs[i][:60]}...")
            print(f"    Predicted:  {predictions[i][:60]}...")
            print(f"    Reference:  {references[i][:60]}...")
            print(f"    BLEU:       {bleu_scores[i]:.4f}")
        
        if results['has_attention']:
            print(f"\n🔗 ATTENTION WEIGHTS: Available for visualization")
        
        return results, error_analysis, (inputs, predictions, references)


def load_model(model_name, checkpoint_path, docstring_vocab, code_vocab, device):
    """Load trained model from checkpoint"""
    
    config = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_layers': 2
    }
    
    # Create model
    if model_name == 'vanilla_rnn':
        model = create_vanilla_rnn_model(
            input_vocab_size=len(docstring_vocab),
            output_vocab_size=len(code_vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            device=device
        )
    elif model_name == 'lstm':
        model = create_lstm_model(
            input_vocab_size=len(docstring_vocab),
            output_vocab_size=len(code_vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            device=device
        )
    elif model_name == 'lstm_attention':
        model = create_lstm_attention_model(
            input_vocab_size=len(docstring_vocab),
            output_vocab_size=len(code_vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            device=device
        )
    elif model_name == 'transformer':
        model = create_transformer_model(
            input_vocab_size=len(docstring_vocab),
            output_vocab_size=len(code_vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            device=device
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded {model_name} from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    # Show training and validation loss
    if 'train_losses' in checkpoint and len(checkpoint['train_losses']) > 0:
        final_train_loss = checkpoint['train_losses'][-1]
        final_val_loss = checkpoint['val_losses'][-1]
        best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
    else:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    return model


def main():
    """Main evaluation function"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ===== CRITICAL: Use saved vocabularies from training =====
    print("\nLoading vocabularies...")
    
    # Auto-detect checkpoint directory
    try:
        import google.colab
        checkpoint_dir = '/content/drive/MyDrive/text2code-seq2seq/checkpoints'
    except:
        checkpoint_dir = 'checkpoints'
    
    docstring_vocab = load_vocab(f'{checkpoint_dir}/docstring_vocab.pkl')
    code_vocab = load_vocab(f'{checkpoint_dir}/code_vocab.pkl')
    
    print(f"Loaded vocabularies:")
    print(f"  Docstring vocab size: {len(docstring_vocab)}")
    print(f"  Code vocab size: {len(code_vocab)}")
    
    # ===== Load test data WITHOUT creating new vocabularies =====
    print("\nLoading test data...")
    
    from datasets import load_dataset
    dataset = load_dataset("Nan-Do/code-search-net-python")
    
    # CodeSearchNet only has 'train' split, so we'll use part of it for testing
    # We'll skip the first 10000 (used for training) and use next 1000 for testing
    print("Dataset splits available:", dataset.keys())
    
    if 'test' in dataset:
        test_raw = dataset['test'].select(range(min(1000, len(dataset['test']))))
    elif 'validation' in dataset:
        test_raw = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    else:
        # Use a different portion of train split for testing
        # Skip first 10000 (training) and next 1000 (validation), use next 1000 for test
        train_split = dataset['train']
        start_idx = 11000  # After training (10000) and validation (1000)
        end_idx = min(start_idx + 1000, len(train_split))
        test_raw = train_split.select(range(start_idx, end_idx))
        print(f"Using train split indices {start_idx} to {end_idx} for testing")
    
    # Process test data using EXISTING vocabularies
    test_data = []
    for example in test_raw:
        docstring = example.get('func_documentation_string', '') or example.get('docstring', '')
        code = example.get('func_code_string', '') or example.get('code', '')
        
        if not docstring or not code:
            continue
        
        # Simple tokenization (same as training)
        docstring = docstring.strip().lower()
        code = code.strip().lower()
        
        # Check length constraints
        if len(docstring.split()) <= 50 and len(code.split()) <= 80:
            test_data.append({
                'docstring': docstring,
                'code': code
            })
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Create dataset with EXISTING vocabularies
    test_dataset = CodeDataset(test_data, docstring_vocab, code_vocab, 50, 80)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Evaluate all models
    models_to_eval = ['vanilla_rnn', 'lstm', 'lstm_attention']
    all_results = {}
    
    for model_name in models_to_eval:
        # Try latest checkpoint first (has all 20 epochs)
        checkpoint_path = f'{checkpoint_dir}/{model_name}_latest.pt'
        
        # Fallback to best if latest doesn't exist
        if not os.path.exists(checkpoint_path):
            checkpoint_path = f'{checkpoint_dir}/{model_name}_best.pt'
        
        if not os.path.exists(checkpoint_path):
            print(f"\nSkipping {model_name} - checkpoint not found")
            continue
        
        # Load model
        model = load_model(model_name, checkpoint_path, docstring_vocab, code_vocab, device)
        
        # Evaluate
        evaluator = Evaluator(model, model_name, device, docstring_vocab, code_vocab)
        results, errors, predictions = evaluator.evaluate(test_loader)
        
        all_results[model_name] = results
        
        # Save results
        with open(f'{checkpoint_dir}/{model_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {checkpoint_dir}/{model_name}_results.json")
    
    # Compare models
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'BLEU':<10} {'Token Acc':<12} {'Exact Match':<12} {'AST Valid':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        bleu = results['bleu_score']['average'] if isinstance(results['bleu_score'], dict) else results['bleu_score']
        token_acc = results.get('token_accuracy', 0)
        exact = results.get('exact_match_rate', 0)
        ast = results.get('ast_valid_rate', 0)
        
        print(f"{model_name:<20} {bleu:<10.4f} {token_acc:<12.2f} {exact:<12.2f} {ast:<12.2f}")
    
    # Detailed comparison
    print(f"\n{'='*70}")
    print("DETAILED ERROR ANALYSIS COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<20} {'Syntax Err':<12} {'Missing Indent':<16} {'Operators':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        if 'error_analysis' in results:
            syntax = results['error_analysis'].get('syntax_errors', 0)
            indent = results['error_analysis'].get('missing_indentation', 0)
            ops = results['error_analysis'].get('incorrect_operators', 0)
            print(f"{model_name:<20} {syntax:<12} {indent:<16} {ops:<12}")
    
    print(f"\n{'='*70}")
    print("LENGTH-BASED PERFORMANCE")
    print(f"{'='*70}\n")
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        bleu_by_len = results.get('bleu_by_docstring_length', {})
        for length_bin, bleu in sorted(bleu_by_len.items()):
            print(f"  {length_bin} tokens: {bleu:.4f}")
    
    # Save comparison
    with open(f'{checkpoint_dir}/model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Full results saved to: {checkpoint_dir}/model_comparison.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()