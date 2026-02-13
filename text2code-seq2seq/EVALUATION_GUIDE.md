# Text2Code Seq2Seq - Complete Guide

## Project Overview

This project implements and compares three Seq2Seq models for code generation from docstrings:
1. **Vanilla RNN** - Baseline RNN-based encoder-decoder
2. **LSTM** - LSTM-based encoder-decoder with better long-range dependency modeling
3. **LSTM + Attention** - LSTM with Bahdanau attention mechanism for improved interpretability

## Key Features

### ✅ Reproducibility
- **Fixed Random Seeds**: All random operations use seed=42 for reproducible results
  - Python's `random` module
  - NumPy operations
  - PyTorch computations
  - CUDA operations (if using GPU)
  - cuDNN deterministic mode enabled

### ✅ Extended Sequence Support
- **Default max docstring length**: 100 tokens (extended from 50)
- **Default max code length**: 150 tokens (extended from 80)
- Configurable per-dataset in training

### ✅ Comprehensive Evaluation Metrics
- **Token-level Accuracy**: Percentage of correctly predicted tokens
- **BLEU Score**: N-gram overlap between generated and reference code
- **Exact Match Accuracy**: Percentage of completely correct outputs
- **Error Analysis**: 
  - Syntax errors (missing colons, unmatched parentheses)
  - Indentation issues
  - Operator/variable mismatches
- **Performance vs Length**: BLEU scores binned by docstring length

### ✅ Attention Analysis (Mandatory)
- **Visualization**: Heatmaps showing attention weights for 3+ test examples
- **Analysis**: Semantic relevance of attended tokens
- **Statistics**: Entropy and diagonal alignment metrics

## File Structure

```
text2code-seq2seq/
├── train.py                      # Main training script
├── evaluate_metrics.py           # Core evaluation metrics module
├── evaluate_all_models.py        # Full evaluation pipeline
├── visualize_attention_final.py  # Attention visualization script
├── data_preprocessing.py         # Data loading and preprocessing
├── models/
│   ├── vanilla_rnn.py
│   ├── lstm_seq2seq.py
│   ├── lstm_attention.py
│   └── transformer.py
├── checkpoints/                  # Saved models and results
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

### Key Dependencies
- torch>=2.0.0
- numpy
- matplotlib
- seaborn
- datasets (Hugging Face)
- nltk (for BLEU score computation)

## Usage Guide

### 1. Train Models with Reproducibility

```bash
# Basic training with default seed (42)
python train.py

# Custom seed for different reproducibility runs
python train.py 123
```

**Configuration in train.py**:
```python
config = {
    "num_train": 10000,
    "num_val": 1500,
    "num_test": 1500,
    "max_docstring_len": 100,  # Extended length
    "max_code_len": 150,
    "embedding_dim": 256,
    "hidden_dim": 256,
    "batch_size": 64,
    "num_epochs": 15,
    "learning_rate": 0.001,
    "num_layers": 2,
    "dropout": 0.5,
    "teacher_forcing_ratio": 0.5,
    "weight_decay": 0.0001
}
```

**Training Features**:
- Automatic checkpoint saving (best and latest)
- Early stopping (patience=5 epochs)
- Gradient clipping (max_norm=1.0)
- Training curve visualization and saving
- Resume from checkpoint if interrupted

### 2. Evaluate All Models

Run comprehensive evaluation on all trained models:

```bash
python evaluate_all_models.py
```

**Output**:
- Metrics for each model (BLEU, token accuracy, exact match)
- Error analysis breakdown
- Performance vs docstring length analysis
- Comparison plots:
  - `model_comparison.png` - Side-by-side metric comparison
  - `performance_vs_length.png` - BLEU vs docstring length
  - `error_analysis.png` - Error type comparison
- JSON results for each model

### 3. Visualize Attention Weights (LSTM+Attention Only)

Generate detailed attention heatmaps for 3 test examples:

```bash
python visualize_attention_final.py
```

**Output**:
- 3 detailed attention visualizations
- Attention analysis for each example:
  - Top attended source tokens
  - Attention entropy (focus measure)
  - Diagonal alignment score
- Example questions answered:
  - Does "maximum" attend to ">" or "max()"?
  - How does attention align with token semantics?

## Evaluation Metrics Details

### Token-level Accuracy
```
Accuracy = (Number of correctly predicted tokens) / (Total non-padding tokens)
```
- Higher is better (0-100%)
- Measures per-token prediction quality

### BLEU Score
```
BLEU = Weighted product of n-gram precisions (n=1 to 4)
```
- Range: 0-1 (converted to 0-100%)
- Measures similarity to reference using SmoothingFunction

### Exact Match Accuracy
```
EM = (Number of perfectly matching outputs) / (Total outputs)
```
- Only counts if entire sequence matches
- Useful for short functions
- More stringent than token accuracy

### Error Analysis
Detected error types:
- **Syntax Errors**: Missing colons, unmatched parentheses/brackets
- **Indentation Issues**: Inconsistent spacing or missing INDENT tokens
- **Operator Errors**: Wrong/missing operators compared to reference

### Performance vs Docstring Length
- Bins docstrings by length (0-10, 10-20, 20-30, etc.)
- Computes average BLEU for each bin
- Shows model degradation on longer sequences
- Attention models typically perform better on longer inputs

## Results Interpretation

### Why Attention Matters

**Without Attention (Vanilla RNN, LSTM)**:
- Fixed-size context vector must encode entire docstring
- Struggles with long sequences
- Cannot selectively focus on relevant parts

**With Attention (LSTM+Attention)**:
- Dynamic context at each decoding step
- Can attend to different docstring parts based on current generation
- Better performance on longer docstrings
- Interpretable: can see what influenced each token

### Key Findings

1. **Model Performance Trend**: Vanilla RNN < LSTM < LSTM+Attention
2. **Length Sensitivity**: Performance degrades with input length; attention models degrade less
3. **Error Types**: Simpler models make more structural errors; attention models make subtler errors

## Checkpoint Management

### Automatic Saving

During training:
- **`{model}_latest.pt`**: Latest checkpoint (updated each epoch)
- **`{model}_best.pt`**: Best checkpoint (lowest validation loss)
- **`{model}_curves.png`**: Training/validation curves

### Checkpoint Contents
```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'train_losses': list,
    'val_losses': list,
    'val_loss': float
}
```

### Resume Training
Automatically implemented. If interrupted:
```bash
python train.py  # Will resume from last checkpoint
```

To start fresh, delete the checkpoint files:
```bash
rm checkpoints/{model}_latest.pt
```

## Reproducibility Checklist

- ✅ Seeds set globally in `set_seed()` function
- ✅ Data splits use fixed seed in dataset loading
- ✅ Dataloader shuffle=True but seed-based
- ✅ Model initialization with consistent random state
- ✅ Same hyperparameters for fair comparison
- ✅ GPU operations deterministic (cuDNN)

## Common Issues

### CUDA Out of Memory
- Reduce `batch_size` in config
- Increase max_docstring_len gradually
- Use smaller `hidden_dim` (e.g., 128)

### Slow Attention Visualization
- Reduce number of examples (change `num_examples=3`)
- Use smaller max length for plotting heatmaps

### Missing Checkpoints
- Check Google Drive path: `/content/drive/MyDrive/text2code-seq2seq/checkpoints`
- Ensure Google Drive is mounted in Colab
- Run training first, then evaluation

## Extending the Code

### Add New Model
1. Create model in `models/new_model.py`
2. Add factory function `create_new_model()`
3. Add to `model_configs` in `train.py` and `evaluate_all_models.py`
4. Update `MODEL_NAMES` in relevant scripts

### Custom Evaluation Metrics
Edit `evaluate_metrics.py` to add new metrics in `EvaluationMetrics` class

### Modify Attention Analysis
Extend `AttentionVisualizer` in `visualize_attention_final.py`

## Citation & References

Dataset: [code-search-net-python](https://huggingface.co/datasets/Nan-Do/code-search-net-python)

Models implemented based on:
- Seq2Seq with Attention: Bahdanau et al. (2015)
- LSTM: Hochreiter & Schmidhuber (1997)

## Author Notes

This implementation prioritizes:
1. **Reproducibility**: Fixed seeds, documented hyperparameters
2. **Evaluation**: Comprehensive metrics beyond just BLEU
3. **Interpretability**: Attention visualization and analysis
4. **Robustness**: Error handling, checkpoint management, longer sequences

All metrics and visualizations follow standard practices in NLP evaluation for code generation tasks.
