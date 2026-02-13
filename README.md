# Text-to-Python Code Generation Using Seq2Seq Models

Complete implementation of three Seq2Seq architectures for code generation from natural language docstrings.

## 🎯 Project Overview

This project implements and compares three recurrent neural network architectures:

1. **Vanilla RNN Seq2Seq** - Baseline model
2. **LSTM Seq2Seq** - Improved long-term dependency handling
3. **LSTM + Attention** - State-of-the-art with Bahdanau attention

### Dataset
- **CodeSearchNet Python** from Hugging Face
- ~10,000 training pairs (docstring → Python code)
- Real-world GitHub code examples

## 📁 Project Structure

```
text2code-seq2seq/
├── data_preprocessing.py      # Dataset loading & tokenization
├── models/
│   ├── __init__.py
│   ├── vanilla_rnn.py         # Model 1: Vanilla RNN
│   ├── lstm_seq2seq.py        # Model 2: LSTM Seq2Seq
│   └── lstm_attention.py      # Model 3: LSTM + Attention
├── train.py                   # Training all models
├── evaluate.py                # BLEU score & metrics
├── visualize_attention.py     # Attention heatmap generation
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd text2code-seq2seq

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for tokenization)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Training

Train all three models:

```bash
python train.py
```

This will:
- Download CodeSearchNet dataset
- Preprocess and tokenize data
- Train Vanilla RNN, LSTM, and LSTM+Attention models
- Save checkpoints in `checkpoints/` directory
- Generate training curve plots

**Training Configuration:**
- Training examples: 10,000
- Validation examples: 1,000
- Test examples: 1,000
- Embedding dimension: 256
- Hidden dimension: 256
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001

**Expected Training Time:**
- Vanilla RNN: ~15-20 minutes
- LSTM: ~20-25 minutes
- LSTM + Attention: ~30-35 minutes

(On CPU. GPU will be much faster)

### 3. Evaluation

Evaluate all trained models:

```bash
python evaluate.py
```

This will:
- Load best checkpoints
- Generate predictions on test set
- Calculate BLEU score, token accuracy, exact match
- Perform error analysis
- Save results to JSON files

**Output:**
```
checkpoints/vanilla_rnn_results.json
checkpoints/lstm_results.json
checkpoints/lstm_attention_results.json
checkpoints/model_comparison.json
```

### 4. Attention Visualization

Generate attention heatmaps (LSTM + Attention model only):

```bash
python visualize_attention.py
```

This will:
- Generate attention heatmaps for 5 test examples
- Analyze attention patterns
- Save visualizations to `attention_plots/`

**Output:**
```
attention_plots/attention_example_1.png
attention_plots/attention_example_2.png
...
```

## 📊 Model Architectures

### 1. Vanilla RNN Seq2Seq

**Encoder:**
- Simple RNN cells
- Fixed-length context vector
- Unidirectional

**Decoder:**
- Simple RNN cells
- Teacher forcing during training

**Limitations:**
- Struggles with long sequences
- Vanishing gradient problem
- Fixed context bottleneck

### 2. LSTM Seq2Seq

**Encoder:**
- LSTM cells with forget gates
- Better gradient flow
- Unidirectional

**Decoder:**
- LSTM cells
- Improved long-term dependencies

**Improvements:**
- ✓ Better handling of long sequences
- ✓ Reduced vanishing gradient
- ✗ Still has fixed context bottleneck

### 3. LSTM + Attention

**Encoder:**
- **Bidirectional LSTM**
- Processes input forward and backward
- Captures richer context

**Decoder:**
- LSTM with Bahdanau attention
- Dynamic context vector
- Attends to relevant input positions

**Attention Mechanism:**
- Additive (Bahdanau) attention
- Learns alignment between input/output
- Interpretable via attention weights

**Improvements:**
- ✓ No fixed context bottleneck
- ✓ Better performance on long sequences
- ✓ Interpretable alignments
- ✓ State-of-the-art results

## 📈 Evaluation Metrics

### 1. BLEU Score
- Measures n-gram overlap with reference
- Industry standard for code generation
- Range: 0-100 (higher is better)

### 2. Token-Level Accuracy
- Percentage of correctly predicted tokens
- Position-aware matching
- Penalizes length mismatches

### 3. Exact Match Accuracy
- Percentage of perfectly generated functions
- Strict metric
- Important for short snippets

### 4. Error Analysis
- **Syntax errors**: Missing keywords, colons, etc.
- **Length mismatches**: Too short/long generations
- **Semantic errors**: Wrong logic but valid syntax

## 🎓 Learning Objectives

### What You'll Learn:

1. **Vanilla RNN Limitations**
   - Vanishing gradients in practice
   - Why long sequences are problematic
   - Baseline performance

2. **LSTM Improvements**
   - How gates help gradient flow
   - Long-term dependency modeling
   - Quantitative improvements

3. **Attention Mechanisms**
   - Breaking the fixed-context bottleneck
   - Alignment learning
   - Interpretability via visualizations

4. **Practical ML Skills**
   - Dataset preprocessing
   - Training loop implementation
   - Evaluation metrics
   - Checkpoint management
   - Visualization techniques

## 📝 Example Usage

### Interactive Testing

```python
from data_preprocessing import load_vocab, sentence_to_indices
from models.lstm_attention import create_lstm_attention_model
import torch

# Load model and vocab
device = torch.device('cpu')
docstring_vocab = load_vocab('checkpoints/docstring_vocab.pkl')
code_vocab = load_vocab('checkpoints/code_vocab.pkl')

model = create_lstm_attention_model(
    len(docstring_vocab), len(code_vocab), 256, 256, 1, device
)
checkpoint = torch.load('checkpoints/lstm_attention_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Test input
docstring = "returns the maximum value in a list"
indices = sentence_to_indices(docstring, docstring_vocab, 50)
src = torch.LongTensor([indices]).to(device)

# Generate
generated, attentions = model.generate(src, 82, 
                                      code_vocab.word2idx['<SOS>'],
                                      code_vocab.word2idx['<EOS>'])

# Decode
output = ' '.join([code_vocab.idx2word[idx.item()] 
                   for idx in generated[0] 
                   if idx.item() != 0 and idx.item() != code_vocab.word2idx['<EOS>']])

print(f"Generated code: {output}")
```

## 🔍 Attention Analysis

The attention visualizations show:

1. **Alignment Quality**
   - Does "maximum" attend to `max()`?
   - Does "list" attend to array operations?

2. **Sequential vs Semantic**
   - Diagonal patterns = sequential copying
   - Scattered patterns = semantic understanding

3. **Common Patterns**
   - Keywords → operators
   - Data types → variable declarations
   - Actions → function calls

## 📊 Expected Results

Based on similar implementations:

| Model | BLEU Score | Token Accuracy | Exact Match |
|-------|-----------|----------------|-------------|
| Vanilla RNN | 15-25 | 40-50% | 5-10% |
| LSTM | 30-40 | 55-65% | 15-20% |
| LSTM + Attention | 45-60 | 70-80% | 25-35% |

*Note: Results vary based on dataset size and hyperparameters*

## 🐛 Troubleshooting

### Common Issues:

1. **Out of Memory**
   ```bash
   # Reduce batch size in train.py
   'batch_size': 16  # instead of 32
   ```

2. **Dataset Download Fails**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/datasets
   python train.py
   ```

3. **CUDA Out of Memory**
   ```bash
   # Use CPU
   # Models will automatically fall back to CPU if CUDA unavailable
   ```

4. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

## 📚 References

- **Attention Mechanism**: [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)
- **Seq2Seq**: [Sutskever et al., 2014](https://arxiv.org/abs/1409.3215)
- **CodeSearchNet**: [Husain et al., 2019](https://arxiv.org/abs/1909.09436)

## 🎯 Assignment Deliverables Checklist

- [x] Vanilla RNN implementation
- [x] LSTM implementation  
- [x] LSTM + Attention implementation
- [x] Training script with loss curves
- [x] Evaluation with BLEU score
- [x] Attention visualization (3+ examples)
- [x] Error analysis
- [x] README with instructions
- [x] Reproducible code

## 🚀 Extensions (Bonus)

1. **Syntax Validation**
   ```python
   import ast
   try:
       ast.parse(generated_code)
       print("Valid Python syntax!")
   except SyntaxError:
       print("Syntax error detected")
   ```

2. **Longer Sequences**
   - Increase `max_docstring_len` and `max_code_len`
   - May need more training data

3. **Transformer Comparison**
   - Implement basic Transformer encoder-decoder
   - Compare with attention-based LSTM

## 📧 Support

For questions or issues:
1. Check this README
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify file paths are correct

## 📜 License

This is an educational project for assignment purposes.

---

**Happy Coding! 🎉**

Built with PyTorch, Hugging Face Datasets


