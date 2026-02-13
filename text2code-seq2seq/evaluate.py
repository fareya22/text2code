"""
Evaluation Script
Calculate BLEU score, token accuracy, and exact match accuracy
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from sacrebleu.metrics import BLEU
from tqdm import tqdm
import json
import os

from data_preprocessing import (
    load_vocab,
    CodeDataset,
    indices_to_sentence,
    load_and_preprocess_data
)
from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model


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
        """Complete evaluation"""
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}\n")
        
        # Generate predictions
        inputs, predictions, references = self.generate_predictions(dataloader)
        
        # Calculate metrics
        bleu_score = self.calculate_bleu(predictions, references)
        token_accuracy = self.calculate_token_accuracy(predictions, references)
        exact_match = self.calculate_exact_match(predictions, references)
        
        # Analyze errors
        errors = self.analyze_errors(inputs, predictions, references)
        
        # Results
        results = {
            'model_name': self.model_name,
            'bleu_score': bleu_score,
            'token_accuracy': token_accuracy,
            'exact_match_accuracy': exact_match,
            'total_examples': len(predictions),
            'error_counts': {
                'syntax_errors': len(errors['syntax_errors']),
                'length_mismatches': len(errors['length_mismatches']),
                'semantic_errors': len(errors['semantic_errors'])
            }
        }
        
        # Print results
        print(f"BLEU Score: {bleu_score:.2f}")
        print(f"Token Accuracy: {token_accuracy:.2f}%")
        print(f"Exact Match Accuracy: {exact_match:.2f}%")
        print(f"\nError Analysis:")
        print(f"  Syntax Errors: {results['error_counts']['syntax_errors']}")
        print(f"  Length Mismatches: {results['error_counts']['length_mismatches']}")
        print(f"  Semantic Errors: {results['error_counts']['semantic_errors']}")
        
        # Show some examples
        print(f"\n{'='*60}")
        print("Sample Predictions:")
        print(f"{'='*60}\n")
        
        for i in range(min(5, len(inputs))):
            print(f"Example {i+1}:")
            print(f"  Input: {inputs[i]}")
            print(f"  Predicted: {predictions[i]}")
            print(f"  Reference: {references[i]}")
            print()
        
        return results, errors, (inputs, predictions, references)


def load_model(model_name, checkpoint_path, docstring_vocab, code_vocab, device):
    """Load trained model from checkpoint"""
    
    config = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_layers': 1
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
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    print(f"{'Model':<20} {'BLEU':<10} {'Token Acc':<12} {'Exact Match':<12}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<20} {results['bleu_score']:<10.2f} "
              f"{results['token_accuracy']:<12.2f} {results['exact_match_accuracy']:<12.2f}")
    
    # Save comparison
    with open(f'{checkpoint_dir}/model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComparison saved to {checkpoint_dir}/model_comparison.json")


if __name__ == "__main__":
    main()