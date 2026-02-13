"""
Advanced Features Implementation:
1. Syntax Validation using Python AST
2. Extend to longer docstrings (already implemented: 100/150 tokens)
3. Transformer model comparison (already in train.py)
4. Code reproducibility (already implemented: set_seed)
"""

import ast
import re
import torch
from typing import Dict, Tuple, List


# ===== 1. SYNTAX VALIDATION USING PYTHON AST =====

class PythonSyntaxValidator:
    """
    Validate generated Python code using AST parsing
    Detects syntax errors, code structure issues
    """
    
    def __init__(self):
        self.syntax_errors = []
        self.warnings = []
    
    def validate(self, code_str: str) -> Dict:
        """
        Validate Python code using AST
        
        Args:
            code_str: Generated Python code as string
        
        Returns:
            dict with validation results
        """
        self.syntax_errors = []
        self.warnings = []
        
        results = {
            'is_valid_python': False,
            'syntax_errors': [],
            'has_function_def': False,
            'has_return': False,
            'indentation_valid': False,
            'docstring_present': False,
            'ast_tree': None,
            'warnings': [],
            'score': 0  # 0-100
        }
        
        # Step 1: Check indentation
        indentation_valid = self._check_indentation(code_str)
        results['indentation_valid'] = indentation_valid
        
        # Step 2: Try to parse with AST
        try:
            tree = ast.parse(code_str)
            results['is_valid_python'] = True
            results['ast_tree'] = tree
            
            # Step 3: Analyze AST
            self._analyze_ast(tree, results)
            
            # Step 4: Calculate score
            results['score'] = self._calculate_score(results)
            
        except SyntaxError as e:
            results['is_valid_python'] = False
            results['syntax_errors'].append({
                'line': e.lineno,
                'offset': e.offset,
                'msg': str(e.msg),
                'text': e.text
            })
        
        except IndentationError as e:
            results['is_valid_python'] = False
            results['syntax_errors'].append({
                'type': 'IndentationError',
                'line': e.lineno,
                'msg': str(e.msg)
            })
        
        except Exception as e:
            results['is_valid_python'] = False
            results['syntax_errors'].append({
                'type': 'ParseError',
                'msg': str(e)
            })
        
        return results
    
    def _check_indentation(self, code_str: str) -> bool:
        """Check if indentation is consistent"""
        lines = code_str.split('\n')
        indent_level = None
        
        for i, line in enumerate(lines):
            if not line.strip():  # Skip empty lines
                continue
            
            leading_spaces = len(line) - len(line.lstrip())
            
            # Check if indentation is multiple of 4
            if leading_spaces % 4 != 0:
                self.warnings.append(
                    f"Line {i+1}: Indentation not multiple of 4 ({leading_spaces} spaces)"
                )
                return False
        
        return True
    
    def _analyze_ast(self, tree: ast.AST, results: Dict) -> None:
        """Analyze AST for code structure"""
        
        for node in ast.walk(tree):
            # Check for function definitions
            if isinstance(node, ast.FunctionDef):
                results['has_function_def'] = True
                
                # Check for docstring in function
                docstring = ast.get_docstring(node)
                if docstring:
                    results['docstring_present'] = True
            
            # Check for return statements
            if isinstance(node, ast.Return):
                results['has_return'] = True
            
            # Check for undefined variables (potential issues)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Note: Simple check - more complex analysis would need symbol table
                pass
    
    def _calculate_score(self, results: Dict) -> int:
        """
        Calculate syntax validity score (0-100)
        
        Args:
            results: Validation results dict
        
        Returns:
            Score 0-100
        """
        score = 0
        
        # Base score: valid Python
        if results['is_valid_python']:
            score += 50
        
        # Additional points
        if results['indentation_valid']:
            score += 15
        
        if results['has_function_def']:
            score += 20
        
        if results['has_return']:
            score += 10
        
        if results['docstring_present']:
            score += 5
        
        return min(score, 100)
    
    def format_error_report(self, results: Dict) -> str:
        """
        Format validation results as readable report
        
        Args:
            results: Validation results dict
        
        Returns:
            Formatted string report
        """
        report = []
        report.append(f"{'='*60}")
        report.append("PYTHON SYNTAX VALIDATION REPORT")
        report.append(f"{'='*60}")
        
        # Validity
        validity = "✓ VALID" if results['is_valid_python'] else "✗ INVALID"
        report.append(f"Status: {validity}")
        report.append(f"Score: {results['score']}/100")
        
        # Syntax errors
        if results['syntax_errors']:
            report.append(f"\n❌ Syntax Errors ({len(results['syntax_errors'])}):")
            for err in results['syntax_errors']:
                if 'line' in err:
                    report.append(f"  Line {err['line']}: {err['msg']}")
                else:
                    report.append(f"  {err['type']}: {err['msg']}")
        
        # Code structure
        report.append(f"\n📋 Code Structure:")
        report.append(f"  Function Definition: {'✓' if results['has_function_def'] else '✗'}")
        report.append(f"  Return Statement: {'✓' if results['has_return'] else '✗'}")
        report.append(f"  Docstring: {'✓' if results['docstring_present'] else '✗'}")
        report.append(f"  Indentation: {'✓' if results['indentation_valid'] else '✗'}")
        
        # Warnings
        if results['warnings']:
            report.append(f"\n⚠ Warnings ({len(results['warnings'])}):")
            for warn in results['warnings']:
                report.append(f"  - {warn}")
        
        report.append(f"{'='*60}\n")
        
        return '\n'.join(report)


# ===== 2. EXTENDED SEQUENCE SUPPORT =====
# Already implemented in train.py:
# - max_docstring_len: 100 tokens (from 50)
# - max_code_len: 150 tokens (from 80)
# This allows handling longer, more complex docstrings and code snippets

EXTENDED_CONFIG = {
    "max_docstring_len": 100,  # Extended from 50
    "max_code_len": 150,        # Extended from 80
    "note": "Supports longer and more complex code generation tasks"
}


# ===== 3. TRANSFORMER MODEL COMPARISON =====
# Already implemented in train.py:
# - create_transformer_model in model construction
# - Included in models_to_train list
# - Generates transformer_best.pt checkpoint
# - Evaluated in evaluate_all_models.py

TRANSFORMER_MODELS = {
    "vanilla_rnn": "Baseline RNN without attention",
    "lstm": "LSTM with gating but no attention",
    "lstm_attention": "LSTM + Bahdanau attention (interpretable)",
    "transformer": "Multi-head self-attention architecture (best for longer sequences)"
}


# ===== 4. CODE REPRODUCIBILITY =====
# Already implemented in train.py and data_preprocessing.py

class ReproducibilityConfig:
    """
    Ensures code reproducibility across runs
    """
    
    DEFAULT_SEED = 42
    
    @staticmethod
    def get_reproducibility_info() -> Dict:
        """Get info about reproducibility setup"""
        return {
            'seed': ReproducibilityConfig.DEFAULT_SEED,
            'implementation': [
                'random.seed(42)',
                'np.random.seed(42)',
                'torch.manual_seed(42)',
                'torch.cuda.manual_seed_all(42)',
                'torch.backends.cudnn.deterministic = True',
                'torch.backends.cudnn.benchmark = False'
            ],
            'usage': 'python train.py 42',
            'verification': 'Same seed → Identical results across runs'
        }


# ===== INTEGRATION: EVALUATE WITH SYNTAX VALIDATION =====

def evaluate_with_syntax_validation(generated_tokens, reference_tokens, vocab_trg):
    """
    Evaluate generated code including syntax validation
    
    Args:
        generated_tokens: List of token IDs
        reference_tokens: List of reference token IDs
        vocab_trg: Target vocabulary
    
    Returns:
        dict with metrics and syntax validation results
    """
    from evaluate_metrics import EvaluationMetrics
    
    metrics = EvaluationMetrics()
    
    # Decode tokens to strings
    gen_tokens = metrics.decode_tokens(torch.tensor(generated_tokens), vocab_trg)
    ref_tokens = metrics.decode_tokens(torch.tensor(reference_tokens), vocab_trg)
    
    gen_code = ' '.join(gen_tokens)
    ref_code = ' '.join(ref_tokens)
    
    # Validate generated code
    validator = PythonSyntaxValidator()
    validation_result = validator.validate(gen_code)
    
    # Compute metrics
    bleu = metrics.compute_bleu(ref_tokens, gen_tokens)
    token_acc = metrics.token_accuracy(
        torch.tensor(generated_tokens),
        torch.tensor(reference_tokens)
    )
    
    return {
        'bleu': bleu,
        'token_accuracy': token_acc,
        'syntax_valid': validation_result['is_valid_python'],
        'syntax_score': validation_result['score'],
        'has_function': validation_result['has_function_def'],
        'has_return': validation_result['has_return'],
        'validation_report': validator.format_error_report(validation_result)
    }


# ===== SUMMARY OF IMPLEMENTATION =====

IMPLEMENTATION_SUMMARY = {
    "1_syntax_validation": {
        "implemented": True,
        "method": "Python AST parsing",
        "detects": [
            "Syntax errors (SyntaxError, IndentationError)",
            "Function definitions",
            "Return statements",
            "Indentation consistency",
            "Code structure validation"
        ],
        "class": "PythonSyntaxValidator",
        "score": "0-100 (validity + structure)"
    },
    
    "2_extended_docstrings": {
        "implemented": True,
        "location": "train.py config",
        "docstring_tokens": 100,  # from 50
        "code_tokens": 150,        # from 80
        "benefit": "Support for longer, more complex tasks"
    },
    
    "3_transformer_model": {
        "implemented": True,
        "location": "train.py model construction + evaluate_all_models.py",
        "architecture": "Multi-head self-attention",
        "comparison": [
            "Vanilla RNN: baseline, struggles with long sequences",
            "LSTM: better long-range, no attention",
            "LSTM+Attention: interpretable, good for medium sequences",
            "Transformer: best for longer sequences, parallel processing"
        ]
    },
    
    "4_reproducibility": {
        "implemented": True,
        "location": "data_preprocessing.py + train.py",
        "method": "Global seed setting (42)",
        "scope": [
            "Python random module",
            "NumPy operations",
            "PyTorch computations",
            "CUDA operations",
            "cuDNN determinism"
        ],
        "usage": "python train.py 42"
    }
}


if __name__ == "__main__":
    # Example: Validate generated code
    
    validator = PythonSyntaxValidator()
    
    # Good code
    good_code = """
def add(a, b):
    '''Returns sum of a and b'''
    return a + b
"""
    
    result_good = validator.validate(good_code)
    print("Good code validation:")
    print(validator.format_error_report(result_good))
    
    # Bad code (syntax error)
    bad_code = """
def add(a, b)
    return a + b
"""
    
    result_bad = validator.validate(bad_code)
    print("\nBad code validation:")
    print(validator.format_error_report(result_bad))
    
    # Print reproducibility info
    print("\nReproducibility Configuration:")
    for key, value in ReproducibilityConfig.get_reproducibility_info().items():
        print(f"  {key}: {value}")
    
    # Print implementation summary
    print("\nImplementation Summary:")
    for feature, details in IMPLEMENTATION_SUMMARY.items():
        print(f"  {feature}: {'✓' if details.get('implemented') else '✗'}")
