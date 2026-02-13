# Quick Start Guide (বাংলা)

## 🚀 দ্রুত শুরু করার উপায়

### ১. সেটআপ করো

```bash
cd text2code-seq2seq

# Dependencies install করো
pip install -r requirements.txt

# NLTK data download করো
python -c "import nltk; nltk.download('punkt')"
```

### ২. প্রথমে Test করো (Optional কিন্তু Recommended)

```bash
python test_models.py
```

এটা verify করবে যে সব models ঠিকমতো কাজ করছে কিনা।

### ৩. Training শুরু করো

```bash
python train.py
```

এটা করবে:
- ✓ CodeSearchNet dataset download করবে
- ✓ তিনটা model train করবে (Vanilla RNN, LSTM, LSTM+Attention)
- ✓ Checkpoints save করবে
- ✓ Loss curves plot করবে

**সময় লাগবে:** CPU তে প্রায় 1-2 ঘন্টা, GPU তে 15-20 মিনিট

### ৪. Evaluation করো

```bash
python evaluate.py
```

এটা দেখাবে:
- ✓ BLEU Score
- ✓ Token Accuracy
- ✓ Exact Match
- ✓ Error Analysis

### ৫. Attention Visualization দেখো

```bash
python visualize_attention.py
```

এটা তৈরি করবে:
- ✓ Attention heatmaps
- ✓ কোন docstring word কোন code token attend করছে তা দেখাবে

## 📁 গুরুত্বপূর্ণ Files

```
text2code-seq2seq/
├── train.py              # ← এটা দিয়ে training করো
├── evaluate.py           # ← এটা দিয়ে evaluation করো
├── visualize_attention.py # ← এটা দিয়ে attention দেখো
├── test_models.py        # ← এটা দিয়ে test করো
│
├── models/
│   ├── vanilla_rnn.py    # Model 1
│   ├── lstm_seq2seq.py   # Model 2
│   └── lstm_attention.py # Model 3
│
├── checkpoints/          # ← Training এর পরে এখানে saves হবে
└── attention_plots/      # ← Visualization এখানে save হবে
```

## ⚙️ Configuration পরিবর্তন করতে চাইলে

`train.py` এর `config` dictionary edit করো:

```python
config = {
    'num_train': 10000,      # ← Training examples (কমাতে পারো: 5000)
    'num_epochs': 20,        # ← Epochs (কমাতে পারো: 10)
    'batch_size': 32,        # ← Batch size (কমাতে পারো: 16)
    'learning_rate': 0.001,
    # ...
}
```

## 🎯 Assignment এর জন্য কী কী লাগবে

1. ✅ তিনটা model trained
2. ✅ Training curves (automatically save হয়)
3. ✅ Evaluation results (JSON files)
4. ✅ Attention visualizations (PNG files)
5. ✅ Source code (already done!)
6. ✅ README (already done!)

## 🐛 সমস্যা হলে

### Memory শেষ হয়ে গেলে:
```python
# train.py এ batch_size কমাও
'batch_size': 16  # 32 এর বদলে
```

### Dataset download না হলে:
```bash
# Cache clear করো
rm -rf ~/.cache/huggingface/datasets
python train.py
```

### Import error হলে:
```bash
pip install -r requirements.txt --force-reinstall
```

## 📊 Expected Results

| Model | BLEU | Token Acc | Exact Match |
|-------|------|-----------|-------------|
| Vanilla RNN | ~20 | ~45% | ~8% |
| LSTM | ~35 | ~60% | ~18% |
| LSTM + Attention | ~50 | ~75% | ~30% |

## 💡 Tips

1. প্রথমে **test_models.py** run করো - এটা নিশ্চিত করবে সব ঠিক আছে
2. Training এ **GPU** use করলে অনেক faster হবে
3. Smaller dataset দিয়ে শুরু করো (5000 examples), পরে বাড়াও
4. Checkpoints **save** হয় automatically - মাঝে training stop করলেও problem নেই

## 🎓 কী শিখবে

- ✅ RNN vs LSTM vs Attention এর practical difference
- ✅ Seq2Seq architecture implementation
- ✅ PyTorch training loop
- ✅ Evaluation metrics (BLEU, accuracy)
- ✅ Attention mechanism visualization
- ✅ Real-world NLP task handling

---

**শুভকামনা! 🚀**

কোন প্রশ্ন থাকলে README.md দেখো বা code এ comments পড়ো।
