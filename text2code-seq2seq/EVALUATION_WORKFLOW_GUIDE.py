"""
Evaluation Workflow Guide
Complete step-by-step guide to evaluate models and generate deliverables
"""

# ==============================================================================
# STEP 1: TRAIN MODELS (Complete Training First)
# ==============================================================================

"""
Run this to train all 3 models:

    python train.py

This will:
  ✓ Train Vanilla RNN
  ✓ Train LSTM
  ✓ Train LSTM+Attention
  ✓ Save checkpoints to checkpoints/
  ✓ Generate training curves
  
Expected output:
  checkpoints/vanilla_rnn_best.pt
  checkpoints/vanilla_rnn_latest.pt
  checkpoints/lstm_best.pt
  checkpoints/lstm_latest.pt
  checkpoints/lstm_attention_best.pt
  checkpoints/lstm_attention_latest.pt
"""

# ==============================================================================
# STEP 2: EVALUATE ALL MODELS
# ==============================================================================

"""
Comprehensive evaluation of all models:

    python evaluate.py

This runs:
  1. BLEU score calculation
  2. Token-level accuracy
  3. Exact match accuracy
  4. Syntax validation (AST)
  5. Error analysis
  6. Length-based analysis

Output files:
  ✓ checkpoints/vanilla_rnn_results.json
  ✓ checkpoints/lstm_results.json
  ✓ checkpoints/lstm_attention_results.json
  ✓ checkpoints/model_comparison.json
"""

# ==============================================================================
# STEP 3: VIEW EVALUATION RESULTS
# ==============================================================================

"""
View results from command line:
"""

import json
from pathlib import Path

def view_results(model_name='lstm_attention'):
    results_file = Path('checkpoints') / f'{model_name}_results.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # BLEU Score
    bleu = results.get('bleu_score', {})
    if isinstance(bleu, dict):
        print(f"BLEU Score:")
        print(f"  Average:    {bleu.get('average', 0):.4f}")
        print(f"  Std Dev:    {bleu.get('std', 0):.4f}")
    else:
        print(f"BLEU Score:   {bleu:.4f}")
    
    # Accuracy Metrics
    print(f"\nAccuracy Metrics:")
    print(f"  Token Accuracy:       {results.get('token_accuracy', 0):.2f}%")
    print(f"  Exact Match Rate:     {results.get('exact_match_rate', 0):.2f}%")
    print(f"  AST Valid Rate:       {results.get('ast_valid_rate', 0):.2f}%")
    
    # Error Analysis
    errors = results.get('error_analysis', {})
    total = errors.get('total_examples', 1)
    print(f"\nError Analysis (out of {total} examples):")
    print(f"  Syntax Errors:        {errors.get('syntax_errors', 0)}")
    print(f"  Missing Indentation:  {errors.get('missing_indentation', 0)}")
    print(f"  Incorrect Operators:  {errors.get('incorrect_operators', 0)}")
    
    # Length Analysis
    bleu_by_length = results.get('bleu_by_docstring_length', {})
    if bleu_by_length:
        print(f"\nBLEU vs Docstring Length:")
        for length in sorted(bleu_by_length.keys(), key=lambda x: float(x)):
            print(f"  {length}-{int(length)+9} tokens:   {bleu_by_length[length]:.4f}")
    
    print(f"\n{'='*70}\n")


# ==============================================================================
# STEP 4: COMPARE ALL MODELS
# ==============================================================================

"""
Compare all models side-by-side:
"""

def compare_models():
    results_dir = Path('checkpoints')
    
    model_names = ['vanilla_rnn', 'lstm', 'lstm_attention']
    all_results = {}
    
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    for model_name in model_names:
        result_file = results_dir / f'{model_name}_results.json'
        if result_file.exists():
            with open(result_file) as f:
                all_results[model_name] = json.load(f)
    
    # Create comparison table
    print(f"{'Model':<20} {'BLEU':<12} {'Token Acc %':<12} {'Exact Match %':<15} {'AST Valid %':<12}")
    print("-" * 70)
    
    for model_name, results in all_results.items():
        bleu = results.get('bleu_score', {})
        if isinstance(bleu, dict):
            bleu_val = bleu.get('average', 0)
        else:
            bleu_val = bleu
        
        token_acc = results.get('token_accuracy', 0)
        exact_match = results.get('exact_match_rate', 0)
        ast_valid = results.get('ast_valid_rate', 0)
        
        print(f"{model_name:<20} {bleu_val:<12.4f} {token_acc:<12.2f} {exact_match:<15.2f} {ast_valid:<12.2f}")
    
    print(f"\n{'='*70}\n")
    
    # Find best model
    best_model = max(all_results.items(), 
                    key=lambda x: x[1].get('bleu_score', {}).get('average', x[1].get('bleu_score', 0)))
    print(f"✅ Best Model: {best_model[0].upper()}")
    print(f"{'='*70}\n")


# ==============================================================================
# STEP 5: VISUALIZE ATTENTION (LSTM+Attention Only)
# ==============================================================================

"""
Generate attention heatmap visualizations:

    python visualize_attention.py

This creates:
  ✓ attention_plots/attention_example_1.png
  ✓ attention_plots/attention_example_2.png
  ✓ ... (multiple examples)

Heatmaps show:
  - X-axis: Source docstring tokens
  - Y-axis: Target code tokens
  - Color: Attention strength
  
Analysis:
  - Does "maximum" attend to "max()" function?
  - Does "list" attend to array operations?
  - Are patterns diagonal (sequential) or scattered (semantic)?
"""

# ==============================================================================
# STEP 6: GENERATE PDF/HTML REPORT
# ==============================================================================

"""
Create comprehensive evaluation report:

    python generate_report.py

This generates:
  ✓ TEXT2CODE_EVALUATION_REPORT.pdf  (if reportlab installed)
  ✓ TEXT2CODE_EVALUATION_REPORT.html (fallback)

Report contains:
  - Executive summary
  - Model comparison table
  - Detailed metrics for each model
  - Error analysis breakdown
  - Length-based performance analysis
  - Methodology explanation
  - Conclusions and recommendations

Install reportlab if needed:
    pip install reportlab
"""

# ==============================================================================
# STEP 7: ANALYZE SPECIFIC PREDICTIONS
# ==============================================================================

"""
Deep dive into model predictions:
"""

def analyze_predictions(model_name='lstm_attention', num_samples=5):
    results_file = Path('checkpoints') / f'{model_name}_results.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    samples = results.get('samples', [])
    
    print(f"\n{'='*70}")
    print(f"SAMPLE PREDICTIONS: {model_name.upper()}")
    print(f"{'='*70}\n")
    
    for i, sample in enumerate(samples[:num_samples]):
        print(f"Example {i+1}:")
        print(f"  📝 Docstring: {sample['docstring'][:80]}...")
        print(f"  🎯 Expected:  {sample['reference'][:80]}...")
        print(f"  🤖 Generated: {sample['generated'][:80]}...")
        print(f"  BLEU: {sample['bleu']:.4f}")
        print(f"  Exact: {'✓' if sample['exact_match'] else '✗'}")
        print()
    
    print(f"{'='*70}\n")


# ==============================================================================
# STEP 8: ERROR TYPE BREAKDOWN
# ==============================================================================

"""
Detailed error analysis:
"""

def analyze_errors(model_name='lstm_attention'):
    results_file = Path('checkpoints') / f'{model_name}_results.json'
    
    if not results_file.exists():
        print(f"❌ Results file not found")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    errors = results.get('error_analysis', {})
    total = errors.get('total_examples', 1)
    
    print(f"\n{'='*70}")
    print(f"ERROR ANALYSIS: {model_name.upper()}")
    print(f"{'='*70}\n")
    
    print(f"Total Examples:          {total}")
    print(f"Syntax Errors:           {errors.get('syntax_errors', 0)} ({errors.get('syntax_errors', 0)/total*100:.1f}%)")
    print(f"Missing Indentation:     {errors.get('missing_indentation', 0)} ({errors.get('missing_indentation', 0)/total*100:.1f}%)")
    print(f"Incorrect Operators:     {errors.get('incorrect_operators', 0)} ({errors.get('incorrect_operators', 0)/total*100:.1f}%)")
    
    print(f"\n{'='*70}\n")


# ==============================================================================
# STEP 9: DELIVERABLES CHECKLIST
# ==============================================================================

"""
Final checklist of all deliverables:

✅ DELIVERABLES CREATED:

1. Source Code Implementations
   ✓ models/vanilla_rnn.py
   ✓ models/lstm_seq2seq.py
   ✓ models/lstm_attention.py
   ✓ models/transformer.py (bonus)

2. Trained Models (Checkpoints)
   ✓ checkpoints/vanilla_rnn_best.pt
   ✓ checkpoints/lstm_best.pt
   ✓ checkpoints/lstm_attention_best.pt
   ✓ checkpoints/model_comparison.json

3. Evaluation Results
   ✓ checkpoints/vanilla_rnn_results.json
   ✓ checkpoints/lstm_results.json
   ✓ checkpoints/lstm_attention_results.json

4. Report (PDF)
   ✓ TEXT2CODE_EVALUATION_REPORT.pdf
   ✓ TEXT2CODE_EVALUATION_REPORT.html

5. Attention Visualizations
   ✓ attention_plots/attention_example_*.png
   ✓ Semantic relevance analysis

6. Documentation
   ✓ README.md
   ✓ METRICS_AND_DELIVERABLES.md
   ✓ REPRODUCIBILITY_GUIDE.md
"""

# ==============================================================================
# STEP 10: COMPLETE WORKFLOW EXAMPLE
# ==============================================================================

"""
Complete workflow from start to finish:

    # 1. Train all models
    python train.py
    
    # 2. Evaluate
    python evaluate.py
    
    # 3. Visualize attention
    python visualize_attention.py
    
    # 4. Generate report
    python generate_report.py
    
    # 5. View results
    python
    >>> from EVALUATION_METRICS_AND_DELIVERABLES import *
    >>> view_results('lstm_attention')
    >>> compare_models()
    >>> analyze_predictions('lstm_attention')
    >>> analyze_errors('lstm_attention')
"""

# ==============================================================================
# RUNNING THE WORKFLOW
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("TEXT2CODE SEQ2SEQ - EVALUATION WORKFLOW")
    print("=" * 70)
    print()
    print("Usage:")
    print("  python EVALUATION_METRICS_AND_DELIVERABLES.py [command]")
    print()
    print("Commands:")
    print("  view <model>           - View results for a model")
    print("  compare                - Compare all models")
    print("  analyze <model>        - Analyze errors for a model")
    print("  predictions <model>    - Show sample predictions")
    print()
    print("Examples:")
    print("  python EVALUATION_METRICS_AND_DELIVERABLES.py view lstm_attention")
    print("  python EVALUATION_METRICS_AND_DELIVERABLES.py compare")
    print()
    print("=" * 70)
    
    if len(sys.argv) < 2:
        compare_models()
    elif sys.argv[1] == 'view':
        model = sys.argv[2] if len(sys.argv) > 2 else 'lstm_attention'
        view_results(model)
    elif sys.argv[1] == 'compare':
        compare_models()
    elif sys.argv[1] == 'analyze':
        model = sys.argv[2] if len(sys.argv) > 2 else 'lstm_attention'
        analyze_errors(model)
    elif sys.argv[1] == 'predictions':
        model = sys.argv[2] if len(sys.argv) > 2 else 'lstm_attention'
        analyze_predictions(model)
    else:
        print(f"Unknown command: {sys.argv[1]}")
