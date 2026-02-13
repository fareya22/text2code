"""
Reproducibility Verification Script
Tests whether training produces identical results with the same seed
"""

import torch
import json
import os
from pathlib import Path

def verify_reproducibility(checkpoint_dir='checkpoints', seed=42):
    """
    Verify reproducibility by checking checkpoint metadata
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        seed: Expected seed value
    """
    print("=" * 60)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 60)
    
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    # Find all .pt files
    checkpoints = list(checkpoint_dir.glob('*_best.pt'))
    
    if not checkpoints:
        print(f"⚠ No checkpoints found in {checkpoint_dir}")
        return False
    
    all_valid = True
    
    for checkpoint_path in sorted(checkpoints):
        print(f"\n📄 Checking: {checkpoint_path.name}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if seed is saved
            if 'seed' not in checkpoint:
                print("  ⚠ WARNING: Seed not saved in this checkpoint")
                all_valid = False
            else:
                checkpoint_seed = checkpoint['seed']
                print(f"  ✓ Seed: {checkpoint_seed}")
                
                if checkpoint_seed != seed:
                    print(f"  ⚠ Expected seed {seed}, got {checkpoint_seed}")
                    all_valid = False
            
            # Display checkpoint info
            print(f"  ✓ Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  ✓ Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
            print(f"  ✓ Model State: {len(checkpoint['model_state_dict'])} tensors")
            print(f"  ✓ Optimizer State: {len(checkpoint['optimizer_state_dict'])} entries")
            
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
            all_valid = False
    
    print("\n" + "=" * 60)
    
    if all_valid:
        print("✅ ALL CHECKS PASSED - Reproducibility enabled!")
    else:
        print("⚠ SOME CHECKS FAILED - Review warnings above")
    
    print("=" * 60)
    
    return all_valid


def compare_training_logs(log1, log2):
    """
    Compare training logs from two runs with same seed
    
    Args:
        log1, log2: Paths to training log files or checkpoint directories
    """
    print("\n" + "=" * 60)
    print("COMPARING TWO TRAINING RUNS")
    print("=" * 60)
    
    try:
        with open(log1 if isinstance(log1, str) else log1, 'r') as f:
            data1 = json.load(f)
        
        with open(log2 if isinstance(log2, str) else log2, 'r') as f:
            data2 = json.load(f)
        
        # Compare losses
        if 'train_losses' in data1 and 'train_losses' in data2:
            losses1 = data1['train_losses']
            losses2 = data2['train_losses']
            
            max_epochs = min(len(losses1), len(losses2))
            all_match = True
            
            for epoch in range(max_epochs):
                diff = abs(losses1[epoch] - losses2[epoch])
                if diff > 1e-6:  # Allow small floating point differences
                    print(f"Epoch {epoch}: Loss1={losses1[epoch]:.6f}, Loss2={losses2[epoch]:.6f}, Diff={diff:.6e}")
                    all_match = False
            
            if all_match:
                print(f"✅ All {max_epochs} epochs have identical losses!")
            else:
                print("⚠ Losses differ between runs")
                
        else:
            print("ℹ Log files don't contain training losses")
            
    except Exception as e:
        print(f"❌ Error comparing logs: {e}")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify reproducibility settings')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42, help='Expected seed')
    parser.add_argument('--compare', nargs=2, help='Compare two checkpoint files')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_training_logs(args.compare[0], args.compare[1])
    else:
        verify_reproducibility(args.checkpoint_dir, args.seed)
