# Colab Execution Guide - Step by Step

এই guide অনুসরণ করে Colab-এ সব কিছু run করো।

## Step 1: Environment Setup (Colab শুরুতে একবার)

```python
# Install dependencies
!pip install -q torch nltk datasets seaborn matplotlib numpy

# Download NLTK data for BLEU score
import nltk
nltk.download('averaged_perceptron_tagger')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone/Upload your project
import os
os.chdir('/content')
# If using git:
# !git clone https://github.com/your-repo/text2code-seq2seq.git
# Or manually upload the folder

cd text2code-seq2seq
```

## Step 2: Training with Reproducibility

```bash
# Reproducible training with default seed (42)
python train.py

# Or with custom seed for comparison
python train.py 123
```

**Expected Output**:
```
Using seed: 42
Using device: cuda
Loading dataset from Hugging Face...
Train: 10000, Val: 1500, Test: 1500

============================================================
Training vanilla_rnn
============================================================
Epoch 1/15
Training vanilla_rnn: 100%|██████| 157/157 [00:45<00:00, 3.45it/s]
Train Loss: 6.2341 | Val Loss: 5.8234
✓ Best model updated!
...
```

**Training Time Estimate**:
- Vanilla RNN: ~2-3 minutes per epoch
- LSTM: ~3-4 minutes per epoch
- LSTM+Attention: ~5-6 minutes per epoch
- Total: ~1.5-2 hours for all 3 models, 15 epochs each

**Checkpoints Saved to**:
```
/content/drive/MyDrive/text2code-seq2seq/checkpoints/
├── vanilla_rnn_best.pt
├── vanilla_rnn_latest.pt
├── lstm_best.pt
├── lstm_latest.pt
├── lstm_attention_best.pt
├── lstm_attention_latest.pt
├── docstring_vocab.pkl
├── code_vocab.pkl
└── *.png (training curves)
```

## Step 3: Full Model Evaluation

```bash
# Evaluate all models on test set
python evaluate_all_models.py
```

**Expected Output**:
```
======================================================================
Evaluating vanilla_rnn...
======================================================================
Training vanilla_rnn...
BLEU Score:              0.2341 (±0.1523)
Token Accuracy:          35.42%
Exact Match Accuracy:    2.34%

Error Analysis:
  Syntax Errors:         145
  Missing Indentation:   78
  Incorrect Operators:   203
======================================================================

[Comparison plot visualization]
[Performance vs length plot]
[Error analysis plot]

Evaluation pipeline complete!
```

**Output Files**:
```
checkpoints/
├── vanilla_rnn_evaluation.json
├── lstm_evaluation.json
├── lstm_attention_evaluation.json
├── model_comparison.png          ← Key visualization
├── performance_vs_length.png     ← Key visualization
├── error_analysis.png            ← Key visualization
└── evaluation_summary.json
```

## Step 4: Attention Visualization (LSTM+Attention Only)

```bash
# Visualize attention weights for 3 test examples
python visualize_attention_final.py
```

**Expected Output**:
```
===========================================================================
ATTENTION VISUALIZATION
===========================================================================

======================================================================
Example 1
======================================================================

Docstring (input):
  returns list of integers between min and max values

Reference (expected):
  def get_range ( min_val , max_val ) : return [ i for i in range ( ...

Generated (model output):
  def range_list ( start , end ) : return [ i for i in range ( start , ...

📊 Attention Analysis:
  Top attended source tokens:
    - 'list': 0.234
    - 'integers': 0.189
    - 'between': 0.145
  Attention entropy (lower=more focused): 2.145
  Diagonal alignment score: 0.456

[Attention heatmap visualization]
```

**Output Files**:
```
checkpoints/attention_visualizations/
├── attention_example_1.png  ← Heatmap visualization
├── attention_example_2.png
└── attention_example_3.png
```

## Step 5: View Results Summary

```python
# Load and print evaluation summary
import json

with open('/content/drive/MyDrive/text2code-seq2seq/checkpoints/evaluation_summary.json') as f:
    summary = json.load(f)

print("="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

for model_name, results in summary['results'].items():
    print(f"\n{model_name.upper()}:")
    print(f"  BLEU Score:       {results['bleu']:.4f}")
    print(f"  Token Accuracy:   {results['token_accuracy']:.2f}%")
    print(f"  Exact Match:      {results['exact_match']:.2f}%")
    print(f"  Examples Tested:  {results['num_examples']}")
```

## Step 6: Download Results

```bash
# All results are automatically saved to Google Drive
# Download locally (in Colab):
!zip -r -q evaluation_results.zip /content/drive/MyDrive/text2code-seq2seq/checkpoints/

# Then download evaluation_results.zip from Files panel
```

## Resuming Training (If Interrupted)

```bash
# Just run training again - checkpoints auto-load!
python train.py

# Output will show:
# Loading checkpoint...
# Resuming from epoch 11
```

To force fresh training (discard previous checkpoints):

```bash
# Delete checkpoints
import os
import shutil
checkpoint_dir = '/content/drive/MyDrive/text2code-seq2seq/checkpoints'
for f in os.listdir(checkpoint_dir):
    if '_latest.pt' in f or '_best.pt' in f:
        os.remove(os.path.join(checkpoint_dir, f))

# Then run training fresh
!python train.py
```

## Expected Results Summary

### Model Performance Ranking

```
BLEU Score:
  1. LSTM+Attention: ~0.35-0.42
  2. LSTM:           ~0.28-0.35
  3. Vanilla RNN:    ~0.20-0.28

Token Accuracy:
  1. LSTM+Attention: ~45-55%
  2. LSTM:           ~38-45%
  3. Vanilla RNN:    ~30-38%

Exact Match (harder metric):
  1. LSTM+Attention: ~5-8%
  2. LSTM:           ~2-4%
  3. Vanilla RNN:    ~0-2%
```

### Key Observations

1. **Attention Helps Significantly**: +30% better BLEU than vanilla RNN
2. **Longer Sequences**: LSTM+Attention maintains quality better on longer docstrings
3. **Error Patterns**: Attention model makes fewer syntax errors
4. **Interpretability**: Attention weights show semantic alignment

## Visualization Guide

### training_curves.png
- X-axis: Epoch
- Y-axis: Loss
- Shows convergence behavior
- Check for overfitting: if val loss increases while train loss decreases

### model_comparison.png
- 3 subplots: BLEU, Token Accuracy, Exact Match
- Compare models side-by-side
- Look for clear winner (usually LSTM+Attention)

### performance_vs_length.png
- X-axis: Docstring length (tokens)
- Y-axis: BLEU score
- Shows robustness to longer sequences
- Attention should be flatter (less degradation)

### attention_example_*.png
- X-axis: Input docstring tokens
- Y-axis: Generated code tokens
- Color intensity: Attention weight (0=white, 1=dark red)
- Red cells = model attended to that word

## Troubleshooting

### Out of Memory Error

```python
# Reduce batch size in train.py:
config['batch_size'] = 32  # from 64

# Or reduce max lengths:
"max_docstring_len": 60,   # from 100
"max_code_len": 100,       # from 150
```

### Slow Evaluation

```python
# In evaluate_all_models.py, reduce test examples:
results = evaluate_model(
    model, test_loader, ...,
    max_examples=100  # instead of None (all)
)
```

### NLTK Data Missing

```python
import nltk
nltk.download('averaged_perceptron_tagger')
```

### Drive Mount Issues

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Key Files to Review

For **Assignment Submission**:

1. **Source Code**:
   - `train.py` - Training with reproducibility
   - `models/*.py` - Model implementations
   - `data_preprocessing.py` - Data handling
   - `evaluate_metrics.py` - Evaluation framework
   - `visualize_attention_final.py` - Attention visualization

2. **Results & Visualizations**:
   - `model_comparison.png` - Metric comparison
   - `performance_vs_length.png` - Robustness analysis
   - `error_analysis.png` - Error breakdown
   - `attention_example_*.png` - Attention heatmaps

3. **Reports**:
   - `evaluation_summary.json` - Numerical results
   - `*_evaluation.json` - Per-model detailed results

4. **Documentation**:
   - `EVALUATION_GUIDE.md` - Complete reference
   - `README.md` - Quick start

## Summary Checklist

- ✅ Fixed seeds (reproducibility)
- ✅ Extended max docstring length (100 tokens)
- ✅ Token accuracy computed
- ✅ BLEU scores computed
- ✅ Exact match accuracy computed
- ✅ Error analysis (syntax, indentation, operators)
- ✅ Performance vs length analysis
- ✅ Attention visualization (3+ examples)
- ✅ Model comparison plots
- ✅ Checkpoint management (resume capability)

All requirements met! ✓
