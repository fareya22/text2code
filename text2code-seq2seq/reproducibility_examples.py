"""
Example: How to Use Reproducibility in Your Code
Demonstrates best practices for reproducible deep learning
"""

import torch
from data_preprocessing import set_seed

# ============================================================
# BASIC EXAMPLE
# ============================================================

def basic_reproducible_training():
    """Minimal example showing reproducibility setup"""
    
    # Step 1: SET SEED FIRST (before any random operations)
    seed = 42
    set_seed(seed)
    
    # Step 2: Now all random operations are reproducible
    model = torch.nn.LSTM(10, 20, batch_first=True)
    
    # Step 3: Use DataLoader with proper settings
    from torch.utils.data import DataLoader, TensorDataset
    
    X = torch.randn(100, 5, 10)
    y = torch.randn(100, 5, 20)
    dataset = TensorDataset(X, y)
    
    # Key settings for reproducibility
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,           # Shuffle data
        num_workers=0,          # No multi-processing ← IMPORTANT
        drop_last=True          # Drop incomplete batches
    )
    
    # Step 4: Run training
    # Your training code here
    print("✓ Training setup is reproducible!")


# ============================================================
# EXPERIMENT TRACKING EXAMPLE
# ============================================================

def experiment_with_seed_tracking():
    """Example with proper experiment tracking"""
    
    import json
    from pathlib import Path
    
    # Define experiment
    experiment_config = {
        'seed': 42,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'model': 'LSTM',
        'hardware': 'CUDA',
    }
    
    # Create experiment directory
    exp_dir = Path(f"experiments/exp_seed{experiment_config['seed']}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    print(f"✓ Experiment saved to {exp_dir}")
    print(f"✓ Config: {json.dumps(experiment_config, indent=2)}")


# ============================================================
# COMPARING RUNS EXAMPLE
# ============================================================

def comparing_runs_with_same_seed():
    """How to verify results are reproducible"""
    
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    def run_training(seed, output_file):
        """Run training and save results"""
        set_seed(seed)
        
        # Create dummy data
        X = torch.randn(100, 10)
        y = torch.randn(100)
        dataset = TensorDataset(X, y)
        
        loader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        # Simulate training
        losses = []
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        for epoch in range(3):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                output = model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.6f}")
        
        # Save results
        torch.save({'losses': losses}, output_file)
        return losses
    
    print("\n🔄 Run 1 with seed=42:")
    losses1 = run_training(42, 'run1_seed42.pt')
    
    print("\n🔄 Run 2 with seed=42 (should be identical):")
    losses2 = run_training(42, 'run2_seed42.pt')
    
    print("\n🔄 Run 3 with seed=99 (should be different):")
    losses3 = run_training(99, 'run3_seed99.pt')
    
    # Compare results
    print("\n📊 RESULTS:")
    print(f"Run 1 vs Run 2 (both seed=42): {'✅ IDENTICAL' if losses1 == losses2 else '❌ DIFFERENT'}")
    print(f"Run 1 vs Run 3 (seed=42 vs 99): {'❌ DIFFERENT' if losses1 != losses3 else '✅ IDENTICAL'}")
    
    # Cleanup
    import os
    os.remove('run1_seed42.pt')
    os.remove('run2_seed42.pt')
    os.remove('run3_seed99.pt')


# ============================================================
# BEST PRACTICES CHECKLIST
# ============================================================

REPRODUCIBILITY_CHECKLIST = """
✅ REPRODUCIBILITY BEST PRACTICES CHECKLIST

Before Training:
□ Import set_seed from data_preprocessing
□ Call set_seed(your_seed) at the very beginning
□ Set num_workers=0 in all DataLoaders
□ Set drop_last=True in training DataLoader

During Training:
□ Save seed in checkpoints
□ Log hyperparameters and seed
□ Use deterministic algorithms (already configured)

After Training:
□ Verify checkpoint contains seed
□ Test reproducibility with same seed
□ Document seed in paper/report
□ Compare with other researchers using same config

Tools Available:
□ verify_reproducibility.py - Check checkpoint metadata
□ REPRODUCIBILITY_GUIDE.md - Complete documentation
□ This file - Code examples
"""


if __name__ == "__main__":
    import sys
    
    print(REPRODUCIBILITY_CHECKLIST)
    
    print("\n" + "=" * 60)
    print("RUNNING EXAMPLES")
    print("=" * 60)
    
    print("\n1️⃣  BASIC EXAMPLE:")
    basic_reproducible_training()
    
    print("\n2️⃣  EXPERIMENT TRACKING:")
    experiment_with_seed_tracking()
    
    print("\n3️⃣  COMPARING RUNS:")
    comparing_runs_with_same_seed()
    
    print("\n" + "=" * 60)
    print("Examples completed! See code comments for explanations.")
    print("=" * 60)
