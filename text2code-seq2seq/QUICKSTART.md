# Quick Start Guide

This guide will help you get started with the Text-to-Code Generation project in **5 simple steps**.

## ⚡ Super Quick Start (One Command)

```bash
# Run everything automatically
./run.sh all
```

This will:
1. ✓ Install dependencies
2. ✓ Train all 4 models
3. ✓ Evaluate on test set
4. ✓ Generate attention visualizations

**Time Required:** ~3-5 hours (GPU) or ~12-16 hours (CPU)

---

## 📝 Step-by-Step Guide

### Step 1: Setup Environment (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd text2code-seq2seq

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**✓ Verify installation:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Step 2: Train Models (2-4 hours)

```bash
# Train all models with seed 42
python train.py 42
```

**What happens:**
- Downloads CodeSearchNet dataset automatically (~500MB)
- Trains 4 models sequentially:
  1. Vanilla RNN (~30 min)
  2. LSTM (~45 min)
  3. LSTM+Attention (~60 min)
  4. Transformer (~90 min)
- Saves checkpoints in `checkpoints/`
- Generates training curves in `checkpoints/*_curves.png`

**Monitor progress:**
- Watch console output for epoch-by-epoch progress
- Training and validation loss will be displayed
- Best models are automatically saved

**💡 Tip:** You can stop and resume training anytime. The script automatically loads the latest checkpoint.

---

### Step 3: Evaluate Models (10-15 minutes)

```bash
# Evaluate all trained models
python evaluate.py
```

**What happens:**
- Loads trained models from `checkpoints/`
- Tests on 1500 test examples
- Calculates metrics:
  * Token Accuracy
  * BLEU Score
  * Exact Match
  * Syntax Validation
- Generates performance vs length analysis
- Saves results in `results/`

**Output files:**
- `results/metrics.json` - All metrics in JSON
- `results/performance_vs_length.png` - Graph
- `results/*_samples.txt` - Sample predictions

**💡 Tip:** Check `results/metrics.json` for detailed comparison.

---

### Step 4: Visualize Attention (2-3 minutes)

```bash
# Generate attention heatmaps
python visualize_attention.py
```

**What happens:**
- Loads LSTM+Attention model
- Generates attention heatmaps for 5 examples
- Analyzes attention patterns
- Saves visualizations in `results/attention_viz/`

**Output files:**
- `attention_example_1.png` through `attention_example_5.png`
- Console analysis of attention patterns

**💡 Tip:** Look for semantic alignments like "maximum" → "max()"

---

### Step 5: Create Report (30-60 minutes)

Use the generated results to create your PDF report:

**Include:**
1. **Training Curves** - From `checkpoints/*_curves.png`
2. **Metrics Table** - From `results/metrics.json`
3. **Performance Graph** - From `results/performance_vs_length.png`
4. **Attention Heatmaps** - From `results/attention_viz/`
5. **Sample Outputs** - From `results/*_samples.txt`
6. **Error Analysis** - Syntax errors, common mistakes

**Report Template Structure:**
```
1. Introduction & Problem Statement
2. Model Architectures
   - Vanilla RNN
   - LSTM
   - LSTM with Attention
   - Transformer (bonus)
3. Training Setup
   - Dataset
   - Hyperparameters
   - Training procedure
4. Results
   - Training/Validation curves
   - Metrics comparison table
   - Performance vs docstring length
5. Attention Analysis
   - Heatmaps with explanations
   - Semantic alignment examples
6. Error Analysis
   - Syntax errors
   - Common mistakes
   - Model limitations
7. Conclusion & Future Work
```

---

## 🎯 Expected Timeline

| Task | Time (GPU) | Time (CPU) |
|------|------------|------------|
| Setup | 5 min | 5 min |
| Training | 3-4 hours | 12-16 hours |
| Evaluation | 10 min | 20 min |
| Visualization | 2 min | 3 min |
| Report Writing | 1 hour | 1 hour |
| **Total** | **~5 hours** | **~14 hours** |

---

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size in train.py
# Line 149: change batch_size from 64 to 32 or 16
```

### Issue 2: Dataset Download Fails
```bash
# Solution: Download manually
python -c "from datasets import load_dataset; load_dataset('Nan-Do/code-search-net-python')"
```

### Issue 3: Import Errors
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue 4: Slow Training on CPU
```bash
# Solution: Reduce dataset size in train.py
# Line 147-149: Change num_train from 10000 to 5000
```

---

## 📊 Checking Results

### Quick Metrics Check
```bash
# View overall metrics
cat results/metrics.json | python -m json.tool

# View sample outputs
cat results/vanilla_rnn_samples.txt | head -50
```

### View Training Curves
```bash
# On Linux/Mac
open checkpoints/vanilla_rnn_curves.png
open checkpoints/lstm_curves.png
open checkpoints/lstm_attention_curves.png

# On Windows
start checkpoints/vanilla_rnn_curves.png
```

### View Attention Heatmaps
```bash
# On Linux/Mac
open results/attention_viz/attention_example_1.png

# On Windows
start results\attention_viz\attention_example_1.png
```

---

## 🚀 Advanced Usage

### Train Individual Models
```bash
# Edit train.py line 206-212 to comment out unwanted models
python train.py
```

### Custom Configuration
```bash
# Edit config dictionary in train.py (lines 147-160)
# Adjust: batch_size, num_epochs, learning_rate, etc.
python train.py
```

### Resume Training
```bash
# Training automatically resumes from latest checkpoint
python train.py
```

### Different Random Seed
```bash
# Train with different seed for variance analysis
python train.py 123
```

---

## 🎓 Understanding Results

### Good BLEU Scores
- Vanilla RNN: 10-20
- LSTM: 20-35
- LSTM+Attention: 35-50
- Transformer: 40-55

### Good Token Accuracy
- Vanilla RNN: 30-40%
- LSTM: 40-50%
- LSTM+Attention: 50-60%
- Transformer: 55-65%

### What Low Scores Mean
- <10 BLEU: Model needs more training or better hyperparameters
- <30% Token Acc: Check data preprocessing and model implementation
- <5% Exact Match: Normal for complex code generation

---

## 📞 Need Help?

1. **Check logs:** Look at console output for error messages
2. **Check files:** Verify checkpoints and results were created
3. **Reduce scale:** Try with smaller dataset first (5000 samples)
4. **Ask for help:** Open a GitHub issue with error details

---

## ✅ Checklist for Submission

- [ ] All 4 models trained successfully
- [ ] Checkpoints saved in `checkpoints/`
- [ ] Evaluation completed
- [ ] `results/metrics.json` exists
- [ ] Training curves generated
- [ ] Performance vs length graph created
- [ ] Attention heatmaps (at least 3) generated
- [ ] Sample outputs saved
- [ ] Report PDF created in `report/`
- [ ] README.md updated with GitHub URL
- [ ] Code cleaned and organized
- [ ] GitHub repository created
- [ ] All files pushed to GitHub

---

**Good luck! 🚀**
