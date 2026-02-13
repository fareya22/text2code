# QUICK START - Colab Execution

## সরাসরি Colab-এ এই ৩টি command চালাও (যথাক্রমে):

### **Step 1: Training**
```bash
python train.py
```

**কী হবে**: 
- Vanilla RNN, LSTM, LSTM+Attention, Transformer train হবে
- 15 epochs প্রতিটি model
- প্রতিটি epoch এর training/validation curves save হবে
- **Checkpoints**: `/checkpoints/{model}_best.pt` এবং `_latest.pt`
- **Training time**: ~1.5-2 ঘণ্টা

---

### **Step 2: Full Evaluation**
```bash
python evaluate_all_models.py
```

**কী হবে**:
- সব models evaluate হবে test set-এ
- **3টি comparison plots** generate হবে:
  - `model_comparison.png` - BLEU, Token Accuracy, Exact Match
  - `performance_vs_length.png` - Length vs performance
  - `error_analysis.png` - Error breakdown

- **JSON results**: 
  - `vanilla_rnn_evaluation.json`
  - `lstm_evaluation.json`
  - `lstm_attention_evaluation.json`
  - `evaluation_summary.json`

**Runtime**: ~5-10 মিনিট

---

### **Step 3: Attention Visualization** (LSTM+Attention only)
```bash
python visualize_attention_final.py
```

**কী হবে**:
- Test set থেকে ৩টি random example pick করবে
- প্রতিটি example এর জন্য:
  - Docstring input দেখাবে
  - Reference code দেখাবে
  - Generated code দেখাবে
  - **Attention heatmap save হবে** (color intensity = attention weight)
  - Top attended words analyze করবে

**Output files**:
```
/checkpoints/attention_visualizations/
├── attention_example_1.png  ← Heatmap visualization
├── attention_example_2.png
└── attention_example_3.png
```

**Runtime**: ~2-3 মিনিট

---

## Expected Output

### Console Output (Training):
```
Using seed: 42
Using device: cuda
Loading dataset from Hugging Face...
Train: 10000, Val: 1500, Test: 1500

============================================================
Training vanilla_rnn
============================================================
Epoch 1/15
Training vanilla_rnn: 100%|██████████| 157/157 [00:45<00:00, 3.45it/s]
Train Loss: 6.2341 | Val Loss: 5.8234
✓ Best model updated!

Epoch 2/15
...
```

### Evaluation Output:
```
======================================================================
Evaluating vanilla_rnn...
======================================================================
✓ vanilla_rnn model created successfully!
Evaluating model...

BLEU Score:              0.2345 (±0.1523)
Token Accuracy:          35.42%
Exact Match Accuracy:    2.34%

Error Analysis (out of 500 examples):
  Syntax Errors:         145
  Missing Indentation:   78
  Incorrect Operators:   203
======================================================================

[Comparison plot visualization]
[Performance vs length plot]
[Error analysis plot]
```

### Attention Visualization Output:
```
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

[Heatmap visualization with color intensity = attention weight]
```

---

## Sample Results (Expected)

```
┌─────────────────┬──────────┬────────────┬─────────────┐
│ Model           │ BLEU ↑   │ Token Acc  │ Exact Match │
├─────────────────┼──────────┼────────────┼─────────────┤
│ Vanilla RNN     │ 0.230    │ 35.2%      │ 1.5%        │
│ LSTM            │ 0.320    │ 42.1%      │ 3.2%        │
│ LSTM+Attention  │ 0.420    │ 51.3%      │ 6.8%        │
└─────────────────┴──────────┴────────────┴─────────────┘
```

---

## Files Required in Checkpoints Before Step 2 & 3

After training (Step 1) complete, you should have:
```
/content/drive/MyDrive/text2code-seq2seq/checkpoints/
├── vanilla_rnn_best.pt           ← Required
├── lstm_best.pt                  ← Required
├── lstm_attention_best.pt        ← Required (for visualization)
├── docstring_vocab.pkl
├── code_vocab.pkl
└── *.png (training curves)
```

---

## Resume Training (If GPU Crashes)

Just run Step 1 again - it will automatically resume from last checkpoint:
```bash
python train.py  # Resumes from where it stopped
```

Delete checkpoints to start fresh:
```python
import os
checkpoint_dir = '/content/drive/MyDrive/text2code-seq2seq/checkpoints'
for f in os.listdir(checkpoint_dir):
    if '_latest.pt' in f or '_best.pt' in f:
        os.remove(os.path.join(checkpoint_dir, f))
```

---

## Troubleshooting

**Out of Memory:**
- Reduce batch_size in train.py: `"batch_size": 32` (from 64)

**Slow Evaluation:**
- In evaluate_all_models.py, reduce examples: `max_examples=100`

**NLTK Error:**
```python
import nltk
nltk.download('averaged_perceptron_tagger')
```

**Drive Not Mounted:**
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

---

## All Requirements Covered ✅

- ✅ Token-level Accuracy
- ✅ BLEU Score
- ✅ Exact Match Accuracy
- ✅ Syntax Error Detection
- ✅ Indentation Error Detection
- ✅ Operator Error Detection
- ✅ Performance vs Length Analysis
- ✅ Attention Visualization (3+ examples)
- ✅ Model Comparison
- ✅ Reproducibility (seed=42)
- ✅ Extended Lengths (100/150 tokens)
- ✅ Checkpoint Resume

**All done! Just run the 3 commands in order.** 🎉
