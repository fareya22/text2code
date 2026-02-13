"""
Train all Seq2Seq Models: Vanilla RNN, LSTM, LSTM+Attention
Kaggle GPU Compatible
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_preprocessing import (
    load_and_preprocess_data,
    CodeDataset,
    save_vocab,
    load_vocab,
    collate_batch,
    set_seed
)

from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model
from models.transformer import create_transformer_model


# Trainer Class (Same as before)
class Trainer:
    def __init__(self, model, model_name, device, save_dir, seed=42):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        self.seed = seed  # Store seed for reproducibility
        os.makedirs(save_dir, exist_ok=True)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5):
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Training {self.model_name}")
        for batch in progress_bar:
            src = batch['docstring'].to(self.device)
            trg = batch['code'].to(self.device)
            optimizer.zero_grad()
            if 'attention' in self.model_name.lower():
                output, _ = self.model(src, trg, teacher_forcing_ratio)
            elif 'transformer' in self.model_name.lower():
                output = self.model(src, trg, teacher_forcing_ratio=None)
            else:
                output = self.model(src, trg, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)
                if 'attention' in self.model_name.lower():
                    output, _ = self.model(src, trg, teacher_forcing_ratio=0)
                elif 'transformer' in self.model_name.lower():
                    output = self.model(src, trg, teacher_forcing_ratio=None)
                else:
                    output = self.model(src, trg, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                output = output[:,1:].reshape(-1, output_dim)
                trg = trg[:,1:].reshape(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def save_checkpoint(self, epoch, val_loss, optimizer, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'seed': self.seed,  # Save seed for reproducibility
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss
        }
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pt')
        torch.save(checkpoint, latest_path)
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pt')
            torch.save(checkpoint, best_path)

    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.0001, weight_decay=0.0001):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        start_epoch = 0
        best_val_loss = float('inf')
        checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pt')
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Verify seed consistency
            checkpoint_seed = checkpoint.get('seed', None)
            if checkpoint_seed is not None and checkpoint_seed != self.seed:
                print(f"⚠ Warning: Resuming with seed {self.seed}, but checkpoint was created with seed {checkpoint_seed}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['val_loss']
            print(f"Resuming from epoch {start_epoch}")
        print(f"\n{'='*60}\nTraining {self.model_name}\n{'='*60}\n")
        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print("✓ Best model updated!")
            self.save_checkpoint(epoch, val_loss, optimizer, is_best=is_best)
        # Plot curves
        plt.figure(figsize=(10,6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'{self.model_name} Training Curve')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{self.model_name}_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {self.save_dir}")


# ===========================
# Main
# ===========================
def main(seed=42):
    """
    Main training function with reproducibility guarantee
    
    Args:
        seed: Random seed for reproducibility (default: 42)
        
    Reproducibility Features:
    - Sets seed for Python, NumPy, PyTorch (CPU & CUDA)
    - Disables cuDNN non-deterministic algorithms
    - Uses num_workers=0 in DataLoaders
    - Saves seed in checkpoints
    - Sets environment variables (PYTHONHASHSEED, CUDA_LAUNCH_BLOCKING)
    
    Usage:
        python train.py 42           # Use seed 42
        python train.py 123          # Use seed 123
        python train.py              # Default seed 42
    """
    # Set seed for reproducibility
    set_seed(seed)
    print(f"Using seed: {seed}")
    
    SAVE_DIR = "/content/drive/MyDrive/text2code-seq2seq/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    config = {
        
        "num_train": 10000,
        "num_val": 1500,
        "num_test": 1500,
        "max_docstring_len": 100,  # Extended from 50 for longer docstrings
        "max_code_len": 150,        # Extended from 80 for longer code
        "embedding_dim": 256,
        "hidden_dim": 256,
        "batch_size": 64,
        "num_epochs": 15,
        "learning_rate": 0.001,
        "num_layers": 2,
        "dropout": 0.5,
        "clip_grad": 1.0,
        "teacher_forcing_ratio": 0.5,     # ← increased from 0.3 for regularization
        "weight_decay": 0.0001   # ← L2 regularization
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data, val_data, test_data, docstring_vocab, code_vocab = load_and_preprocess_data(
        num_train=config['num_train'],
        num_val=config['num_val'],
        num_test=config['num_test'],
        max_docstring_len=config['max_docstring_len'],
        max_code_len=config['max_code_len']
    )

    save_vocab(docstring_vocab, os.path.join(SAVE_DIR, 'docstring_vocab.pkl'))
    save_vocab(code_vocab, os.path.join(SAVE_DIR, 'code_vocab.pkl'))

    train_dataset = CodeDataset(train_data, docstring_vocab, code_vocab,
                                config['max_docstring_len'], config['max_code_len'])
    val_dataset = CodeDataset(val_data, docstring_vocab, code_vocab,
                              config['max_docstring_len'], config['max_code_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_batch, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_batch, num_workers=0, drop_last=False)

    models_to_train = [
        ('vanilla_rnn', create_vanilla_rnn_model),
        ('lstm', create_lstm_model),
        ('lstm_attention', create_lstm_attention_model),
        ('transformer', create_transformer_model)
    ]

    for model_name, model_factory in models_to_train:
        print(f"\nCreating {model_name} model...")
        if model_name == 'vanilla_rnn':
            model = model_factory(
                input_vocab_size=len(docstring_vocab),
                output_vocab_size=len(code_vocab),
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                device=device
            )
        else:
            model = model_factory(
                input_vocab_size=len(docstring_vocab),
                output_vocab_size=len(code_vocab),
                embedding_dim=config['embedding_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config.get('dropout', 0.0),
                device=device
            )
        print(f"✓ {model_name} model created successfully!")
        trainer = Trainer(model, model_name, device, SAVE_DIR, seed=seed)  # Pass seed
        trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'], learning_rate=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0))

    print("\nALL MODELS TRAINED SUCCESSFULLY!")

if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed=seed)

