"""
Comprehensive evaluation metrics for Seq2Seq models
- Token-level Accuracy
- BLEU Score
- Exact Match Accuracy
- Error Analysis (Syntax, indentation, operators)
- Performance vs Docstring Length
"""

import torch
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re


class EvaluationMetrics:
    """Class for computing evaluation metrics"""
    
    def __init__(self, pad_idx=0, eos_idx=2):
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.smooth_fn = SmoothingFunction().method1
    
    def decode_tokens(self, indices, vocab):
        """Convert token indices to tokens"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        
        tokens = []
        for idx in indices:
            if idx == self.pad_idx or idx == self.eos_idx:
                break
            if idx == 1:  # SOS token
                continue
            token = vocab.itos.get(int(idx), '<UNK>')
            tokens.append(token)
        return tokens
    
    def token_accuracy(self, predictions, targets, pad_idx=None):
        """
        Compute token-level accuracy
        
        Args:
            predictions: Predicted token indices (batch_size, seq_len)
            targets: Target token indices (batch_size, seq_len)
            pad_idx: Padding index to ignore
        
        Returns:
            Token accuracy as percentage
        """
        if pad_idx is None:
            pad_idx = self.pad_idx
        
        mask = targets != pad_idx
        correct = (predictions == targets) & mask
        accuracy = correct.sum().float() / mask.sum().float()
        return accuracy.item() * 100
    
    def compute_bleu(self, reference, hypothesis, max_n=4):
        """
        Compute BLEU score for a single example
        
        Args:
            reference: List of reference tokens (list of strings)
            hypothesis: List of hypothesis tokens (list of strings)
            max_n: Maximum n-gram order
        
        Returns:
            BLEU score (0-1)
        """
        if len(hypothesis) == 0:
            return 0.0
        
        # Create n-grams
        weights = tuple([1.0 / max_n] * max_n)
        try:
            bleu = sentence_bleu(
                [reference],
                hypothesis,
                weights=weights,
                smoothing_function=self.smooth_fn
            )
            return bleu
        except:
            return 0.0
    
    def exact_match(self, reference, hypothesis):
        """
        Check if output exactly matches reference
        
        Args:
            reference: Reference tokens (list of strings)
            hypothesis: Hypothesis tokens (list of strings)
        
        Returns:
            1 if exact match, 0 otherwise
        """
        return 1.0 if reference == hypothesis else 0.0
    
    def analyze_syntax_errors(self, code_tokens):
        """
        Detect common syntax errors in generated code
        
        Returns:
            dict with error counts
        """
        code_str = ' '.join(code_tokens)
        errors = {
            'missing_colon': code_str.count(':') < 1,
            'unmatched_parens': code_str.count('(') != code_str.count(')'),
            'unmatched_brackets': code_str.count('[') != code_str.count(']'),
            'missing_equals': '=' not in code_str and 'return' in code_str,
        }
        return errors
    
    def analyze_indentation_errors(self, code_tokens):
        """Check for indentation-related issues"""
        code_str = ' '.join(code_tokens)
        errors = {
            'has_indent_token': 'INDENT' in code_str,
            'inconsistent_spacing': code_str.count('  ') > len(code_tokens) // 2
        }
        return errors
    
    def analyze_operator_errors(self, reference_tokens, hypothesis_tokens):
        """Compare operators and variables between reference and hypothesis"""
        ref_str = ' '.join(reference_tokens)
        hyp_str = ' '.join(hypothesis_tokens)
        
        # Common operators
        operators = ['+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '=']
        
        errors = {
            'missing_operators': 0,
            'wrong_operators': 0,
        }
        
        for op in operators:
            ref_count = ref_str.count(op)
            hyp_count = hyp_str.count(op)
            if ref_count > hyp_count:
                errors['missing_operators'] += (ref_count - hyp_count)
            elif hyp_count > ref_count:
                errors['wrong_operators'] += (hyp_count - ref_count)
        
        return errors


def evaluate_model(model, dataloader, vocab_src, vocab_trg, device, 
                   has_attention=False, max_examples=None):
    """
    Comprehensive evaluation of model on test set
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        vocab_src: Source vocabulary
        vocab_trg: Target vocabulary
        device: Device (cuda/cpu)
        has_attention: Whether model uses attention
        max_examples: Maximum examples to evaluate
    
    Returns:
        dict with all evaluation metrics
    """
    metrics = EvaluationMetrics()
    model.eval()
    
    all_bleu = []
    all_token_acc = []
    all_exact_match = []
    
    error_analysis = {
        'syntax_errors': 0,
        'missing_indentation': 0,
        'incorrect_operators': 0,
        'total_examples': 0
    }
    
    docstring_lengths = []
    bleu_by_length = {}
    
    samples = []
    
    with torch.no_grad():
        example_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            src = batch['docstring'].to(device)
            trg = batch['code'].to(device)
            
            batch_size = src.shape[0]
            
            # Infer
            if has_attention:
                output, _ = model(src, trg, teacher_forcing_ratio=0)
            else:
                output = model(src, trg, teacher_forcing_ratio=0)
            
            predictions = output.argmax(dim=-1)  # (batch, seq_len)
            
            # Compute metrics for each example in batch
            for i in range(batch_size):
                if max_examples and example_count >= max_examples:
                    break
                
                src_tokens = metrics.decode_tokens(src[i], vocab_src)
                ref_tokens = metrics.decode_tokens(trg[i], vocab_trg)
                pred_tokens = metrics.decode_tokens(predictions[i], vocab_trg)
                
                # Token accuracy
                trg_mask = trg[i] != 0
                pred_mask = predictions[i] != 0
                
                if trg_mask.sum() > 0:
                    correct = (predictions[i] == trg[i]) & trg_mask
                    token_acc = correct.sum().float() / trg_mask.sum().float()
                    all_token_acc.append(token_acc.item() * 100)
                
                # BLEU score
                bleu = metrics.compute_bleu(ref_tokens, pred_tokens)
                all_bleu.append(bleu)
                
                # Exact match
                exact = metrics.exact_match(ref_tokens, pred_tokens)
                all_exact_match.append(exact)
                
                # Error analysis
                error_analysis['total_examples'] += 1
                
                syntax_err = metrics.analyze_syntax_errors(pred_tokens)
                if any(syntax_err.values()):
                    error_analysis['syntax_errors'] += 1
                
                indent_err = metrics.analyze_indentation_errors(pred_tokens)
                if any(indent_err.values()):
                    error_analysis['missing_indentation'] += 1
                
                op_err = metrics.analyze_operator_errors(ref_tokens, pred_tokens)
                if op_err['wrong_operators'] > 0:
                    error_analysis['incorrect_operators'] += 1
                
                # Length analysis
                doc_len = len(src_tokens)
                docstring_lengths.append(doc_len)
                
                # Bin lengths
                length_bin = (doc_len // 10) * 10
                if length_bin not in bleu_by_length:
                    bleu_by_length[length_bin] = []
                bleu_by_length[length_bin].append(bleu)
                
                # Store sample
                samples.append({
                    'docstring': ' '.join(src_tokens),
                    'reference': ' '.join(ref_tokens),
                    'generated': ' '.join(pred_tokens),
                    'bleu': bleu,
                    'exact_match': exact,
                    'token_accuracy': all_token_acc[-1] if all_token_acc else 0,
                    'docstring_length': doc_len
                })
                
                example_count += 1
    
    # Aggregate metrics
    results = {
        'avg_bleu': np.mean(all_bleu) if all_bleu else 0.0,
        'std_bleu': np.std(all_bleu) if all_bleu else 0.0,
        'token_accuracy': np.mean(all_token_acc) if all_token_acc else 0.0,
        'exact_match_rate': np.mean(all_exact_match) if all_exact_match else 0.0,
        'error_analysis': error_analysis,
        'bleu_by_length': {k: np.mean(v) for k, v in bleu_by_length.items()},
        'samples': samples,
        'num_examples': len(samples)
    }
    
    return results


def print_evaluation_report(results, model_name):
    """Print formatted evaluation report"""
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*70}")
    print(f"BLEU Score:              {results['avg_bleu']:.4f} (±{results['std_bleu']:.4f})")
    print(f"Token Accuracy:          {results['token_accuracy']:.2f}%")
    print(f"Exact Match Accuracy:    {results['exact_match_rate']:.2f}%")
    print(f"\nError Analysis (out of {results['error_analysis']['total_examples']} examples):")
    print(f"  Syntax Errors:         {results['error_analysis']['syntax_errors']}")
    print(f"  Missing Indentation:   {results['error_analysis']['missing_indentation']}")
    print(f"  Incorrect Operators:   {results['error_analysis']['incorrect_operators']}")
    print(f"{'='*70}\n")


def save_evaluation_results(results, filepath):
    """Save evaluation results to JSON"""
    import json
    
    # Convert numpy types to native Python types for JSON serialization
    results_copy = results.copy()
    results_copy['avg_bleu'] = float(results_copy['avg_bleu'])
    results_copy['std_bleu'] = float(results_copy['std_bleu'])
    results_copy['token_accuracy'] = float(results_copy['token_accuracy'])
    results_copy['exact_match_rate'] = float(results_copy['exact_match_rate'])
    results_copy['bleu_by_length'] = {
        str(k): float(v) for k, v in results_copy['bleu_by_length'].items()
    }
    
    # Don't save samples (too large)
    if 'samples' in results_copy:
        del results_copy['samples']
    
    with open(filepath, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"Results saved to {filepath}")
