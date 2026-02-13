# Reproducibility Guide - Text2Code Seq2Seq

## Overview

Reproducibility ensures that the same results are obtained when running the same code with the same inputs. This is critical for:
- Validating research results
- Comparing model performances fairly
- Debugging issues
- Collaboration and transparency

## Implementation Details

### 1. **Random Seed Management**

The `set_seed()` function in `data_preprocessing.py` controls randomness across all libraries:

```python
set_seed(seed=42)  # Default seed
```

**What it does:**
- Sets Python's `random` module seed
- Sets NumPy's `np.random` seed  
- Sets PyTorch's CPU seed with `torch.manual_seed()`
- Sets PyTorch's CUDA seed with `torch.cuda.manual_seed_all()`
- Disables cuDNN non-deterministic algorithms
- Disables cuDNN benchmarking optimization
- Sets environment variables:
  - `PYTHONHASHSEED`: Controls hash randomization
  - `CUDA_LAUNCH_BLOCKING`: Ensures synchronous GPU operations

### 2. **DataLoader Configuration**

Both train and validation DataLoaders use reproducible settings:

```python
DataLoader(dataset, 
           batch_size=64,
           shuffle=True,
           num_workers=0,        # ← No multi-processing 
           drop_last=True,       # ← Consistent batch sizes
           collate_fn=collate_batch)
```

**Why these settings matter:**
- `num_workers=0`: Avoids randomness from worker processes
- `drop_last=True` (train): Ensures same batch size every epoch
- `drop_last=False` (val): Preserves all validation samples

### 3. **Checkpoint Management**

Every checkpoint now saves the seed used during training:

```python
checkpoint = {
    'epoch': epoch,
    'seed': self.seed,  # ← Stored for verification
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    ...
}
```

**When resuming training:**
- The saved seed is verified
- A warning is shown if resuming with a different seed
- This ensures training consistency

## Usage

### Running with Default Seed (42)

```bash
python train.py
```

### Running with Custom Seed

```bash
# Seed 123
python train.py 123

# Seed 99
python train.py 99
```

### Reproducing Previous Results

1. Check the checkpoint file to see which seed was used:
   ```python
   checkpoint = torch.load('checkpoints/lstm_best.pt')
   print(f"Original seed: {checkpoint['seed']}")
   ```

2. re-run training with the same seed:
   ```bash
   python train.py 42
   ```

## Factors That Ensure Reproducibility

✅ **Controlled:**
- Model initialization (same seed)
- Data shuffling order (deterministic shuffle)
- Optimizer state (saved & restored)
- PyTorch operations (deterministic mode)

⚠️ **May Still Vary:**
- Different hardware (GPU models compute differently)
- Different CUDA/cuDNN versions
- Different PyTorch versions
- Multi-GPU training (synchronization issues)

## Best Practices

### 1. Always Set Seed Early

```python
from data_preprocessing import set_seed
set_seed(42)
# All model initialization after this line is reproducible
```

### 2. Save Seed with Experiments

Include seed in your experiment logs:
```python
config = {
    'seed': 42,
    'learning_rate': 0.001,
    'batch_size': 64,
    ...
}
```

### 3. Document Your Seeds

When publishing results or sharing code:
```markdown
## Reproducibility
- Seed: 42
- PyTorch version: 2.0.1
- CUDA version: 11.8
- Hardware: NVIDIA A100
```

### 4. Verify Reproducibility

Test that results are reproducible across runs:

```bash
# First run
python train.py 42

# Second run - should give identical loss curves
python train.py 42

# Different seed - should give different results
python train.py 99
```

## Advanced Topics

### Worker Seeds

For multi-worker DataLoaders (if `num_workers > 0`):
```python
def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

DataLoader(..., 
           num_workers=4,
           worker_init_fn=seed_worker)
```

Currently not used to keep implementation simple.

### Distributed Training

For multi-GPU training, additional measures needed:
```python
torch.distributed.init_process_group(
    backend='nccl',
    init_method='env://'
)
set_seed(42)  # Call after initialization
```

## Troubleshooting

### Results Still Don't Match

**Possible causes:**
1. Different PyTorch/CUDA version
2. Different hardware (CPU vs GPU)
3. Seed set at different point in code
4. Floating-point accumulation differences

**Solution:**
- Compare results on same machine
- Check PyTorch version: `torch.__version__`
- Check CUDA version: `torch.cuda.get_device_name(0)`

### Training Slower with Deterministic Mode

**Why:** Deterministic GPU operations are slower
**Solution:** For quick tests, you can disable:
```python
# WARNING: Results no longer reproducible
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

## References

- [PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [CUDA Determinism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)
- [NumPy Random Seed](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html)

## Summary Checklist

- [ ] Always call `set_seed(42)` first
- [ ] Use `num_workers=0` in DataLoaders
- [ ] Save seed in checkpoints
- [ ] Verify on same hardware for true reproducibility
- [ ] Document seed in experiment logs
- [ ] Test reproducibility across multiple runs
