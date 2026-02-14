# Text-to-Code Generation using Seq2Seq Models

Deep Learning project implementing and comparing different sequence-to-sequence architectures for automatic Python code generation from natural language descriptions (docstrings).

## 📋 Project Overview

This project implements and evaluates four neural sequence-to-sequence models for translating natural language function descriptions into Python code:

1. **Vanilla RNN Seq2Seq** - Baseline model with basic RNN encoder-decoder
2. **LSTM Seq2Seq** - Improved model using LSTM cells for better long-term dependencies
3. **LSTM with Attention** - LSTM model with Bahdanau attention mechanism
4. **Transformer** (Bonus) - State-of-the-art transformer-based model

## 🎯 Key Features

- ✅ Complete implementation of 4 different architectures
- ✅ Comprehensive evaluation metrics (BLEU, Token Accuracy, Exact Match, Syntax Validation)
- ✅ Attention visualization with heatmaps
- ✅ Performance analysis vs docstring length
- ✅ Error analysis (syntax errors, indentation mistakes)
- ✅ Reproducible training with seed control
- ✅ Professional documentation and code organization

## 📦 Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM
- 10GB+ free disk space

### Python Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- PyTorch >= 2.0.0
- Hugging Face datasets
- sacrebleu (for BLEU score calculation)
- matplotlib, seaborn (for visualization)
- tqdm (for progress bars)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd text2code-seq2seq
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset

The dataset (CodeSearchNet Python) will be automatically downloaded on first run. No manual download needed!

### 4. Train Models
```bash
# Train all models (Vanilla RNN, LSTM, LSTM+Attention, Transformer)
python train.py

# Or with custom seed for reproducibility
python train.py 42
```

**Training Time (approx):**
- Vanilla RNN: ~30-45 min (GPU) / ~2-3 hours (CPU)
- LSTM: ~45-60 min (GPU) / ~3-4 hours (CPU)
- LSTM+Attention: ~60-90 min (GPU) / ~4-5 hours (CPU)
- Transformer: ~90-120 min (GPU) / ~5-6 hours (CPU)

### 5. Evaluate Models
```bash
# Evaluate all trained models on test set
python evaluate.py
```

**Outputs:**
- `results/metrics.json` - All evaluation metrics
- `results/performance_vs_length.png` - Performance vs docstring length graph
- `results/*_samples.txt` - Sample predictions for each model

### 6. Visualize Attention
```bash
# Generate attention heatmaps (for LSTM+Attention model)
python visualize_attention.py
```

**Outputs:**
- `results/attention_viz/attention_example_*.png`
- Console output with analysis

### 7. Generate PDF Report
```bash
# Generate comprehensive PDF report
python generate_pdf_report.py
```

**Output:** `1331.pdf`

## 📁 Project Structure

```
text2code-seq2seq/
├── models/                          # Model implementations
│   ├── __init__.py
│   ├── vanilla_rnn.py              # Vanilla RNN Seq2Seq
│   ├── lstm_seq2seq.py             # LSTM Seq2Seq
│   ├── lstm_attention.py           # LSTM with Bahdanau Attention
│   └── transformer.py              # Transformer Seq2Seq (bonus)
│
├── train.py                         # Training script for all models
├── evaluate.py                      # Comprehensive evaluation script
├── visualize_attention.py          # Attention visualization script
├── data_preprocessing.py           # Dataset loading and preprocessing
├── utils.py                         # Utility functions
│
├── checkpoints/                     # Saved model checkpoints
│   ├── vanilla_rnn_best.pt
│   ├── vanilla_rnn_latest.pt
│   ├── lstm_best.pt
│   ├── lstm_latest.pt
│   ├── lstm_attention_best.pt
│   ├── lstm_attention_latest.pt
│   ├── transformer_best.pt
│   ├── transformer_latest.pt
│   ├── docstring_vocab.pkl
│   └── code_vocab.pkl
│
├── results/                         # Evaluation results
│   ├── loss_curves/                # Training/validation curves
│   ├── attention_viz/              # Attention heatmaps
│   ├── metrics.json                # All evaluation metrics
│   ├── performance_vs_length.png   # Performance analysis
│   └── *_samples.txt               # Sample outputs
│
├── report/                          # Final report (PDF)
│   └── report.pdf
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── Dockerfile                       # Docker container (optional)
```

## 🔧 Configuration

### Training Configuration (in train.py)

```python
config = {
    "num_train": 10000,           # Training samples
    "num_val": 1500,              # Validation samples
    "num_test": 1500,             # Test samples
    "max_docstring_len": 100,     # Max docstring length
    "max_code_len": 150,          # Max code length
    "embedding_dim": 256,         # Embedding dimension
    "hidden_dim": 256,            # Hidden dimension
    "batch_size": 64,             # Batch size
    "num_epochs": 15,             # Training epochs
    "learning_rate": 0.001,       # Learning rate
    "num_layers": 2,              # LSTM layers
    "dropout": 0.5,               # Dropout rate
    "teacher_forcing_ratio": 0.5, # Teacher forcing
    "weight_decay": 0.0001        # L2 regularization
}
```

### Evaluation Configuration (in evaluate.py)

```python
config = {
    "num_test": 1500,             # Test samples
    "max_docstring_len": 100,     # Max docstring length
    "max_code_len": 150,          # Max generation length
    "batch_size": 32,             # Batch size
}
```

## 📊 Evaluation Metrics

### 1. Token-level Accuracy
Percentage of correctly predicted tokens (excluding padding).

### 2. BLEU Score
N-gram overlap between generated and reference code (using sacrebleu).

### 3. Exact Match Accuracy
Percentage of completely correct outputs.

### 4. Syntax Validation
Percentage of syntactically valid Python code (using AST parser).

### 5. Performance vs Docstring Length
BLEU score analysis across different docstring length buckets.

## � Experimental Results

(Results on 1,500 test samples)

| Model | Token Accuracy (%) | BLEU Score | Syntax Valid (%) |
| :--- | :---: | :---: | :---: |
| **LSTM + Attention** | **12.60%** | **11.98** | 5.0% |
| **LSTM (Seq2Seq)** | 11.31% | 11.07 | 5.0% |
| **Vanilla RNN** | 10.05% | 8.98 | 0.0% |
| **Transformer** | 0.51% | 3.12 | 83.0%* |

\* *Note: Transformer's high syntax validity is due to generating very short, trivial sequences.*

### Key Observations
- **LSTM + Attention** achieved the best performance (highest BLEU & Token Accuracy).
- **Attention** helps align keywords (e.g., "sum" → "+") for better translation.
- **Transformer** struggled with the small dataset (10k samples) due to lack of inductive bias.
- **Vanilla RNN** failed to capture long-range dependencies effectively.

For a detailed analysis, see the generated report: `1331.pdf`.

## 🎨 Attention Visualization

The attention visualization script generates heatmaps showing:
- Which docstring words the model attends to when generating each code token
- Alignment patterns between natural language and code
- Semantic relationships (e.g., "maximum" → "max()")

**Example Analysis:**
```
'max' attends to:
  1. 'maximum' (weight: 0.872)
  2. 'value' (weight: 0.091)
  3. 'returns' (weight: 0.024)
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
config["batch_size"] = 32  # or 16
```

### Dataset Download Issues
```bash
# Manually download dataset
from datasets import load_dataset
dataset = load_dataset("Nan-Do/code-search-net-python")
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## 🔬 Reproducibility

This project ensures reproducibility through:
- ✅ Fixed random seeds (Python, NumPy, PyTorch)
- ✅ Deterministic CUDA operations
- ✅ Seed saved in checkpoints
- ✅ num_workers=0 in DataLoaders
- ✅ Environment variables set (PYTHONHASHSEED, CUDA_LAUNCH_BLOCKING)

To reproduce results:
```bash
python train.py 42  # Use same seed
```


## 📚 References

1. Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
2. Bahdanau et al. (2014) - "Neural Machine Translation by Jointly Learning to Align and Translate"
3. Vaswani et al. (2017) - "Attention Is All You Need"
4. Husain et al. (2019) - "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search"


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** This is an educational project for understanding sequence-to-sequence models. For production code generation, consider using state-of-the-art models like CodeT5, CodeGen, or GitHub Copilot.
