
"""
Training Script for All Seq2Seq Models
Supports: Vanilla RNN, LSTM, LSTM+Attention
Resume Training Supported
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_preprocessing import (
    load_and_preprocess_data,
    CodeDataset,
    save_vocab,
    load_vocab
)

from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model


# ===========================
# Trainer Class
# ===========================

class Trainer:

    def __init__(self, model, model_name, device, save_dir):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []

    # -----------------------
    # Train One Epoch
    # -----------------------
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
            else:
                output = self.model(src, trg, teacher_forcing_ratio)

            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return epoch_loss / len(dataloader)

    # -----------------------
    # Validation
    # -----------------------
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)

                if 'attention' in self.model_name.lower():
                    output, _ = self.model(src, trg, teacher_forcing_ratio=0)
                else:
                    output = self.model(src, trg, teacher_forcing_ratio=0)

                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)
                epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    # -----------------------
    # Save Checkpoint
    # -----------------------
    def save_checkpoint(self, epoch, val_loss, optimizer, is_best=False):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss
        }

        # Save latest (for resume)
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pt')
            torch.save(checkpoint, best_path)

    # -----------------------
    # Training Loop (Resume Supported)
    # -----------------------
    def train(self, train_loader, val_loader, num_epochs, learning_rate=0.001):

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_epoch = 0
        best_val_loss = float('inf')

        checkpoint_path = os.path.join(self.save_dir, f'{self.model_name}_latest.pt')

        # ===== Resume Logic =====
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            best_val_loss = checkpoint['val_loss']

            print(f"Resuming from epoch {start_epoch}")

        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")

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

        self.plot_training_curves()

    # -----------------------
    # Plot Curves
    # -----------------------
    def plot_training_curves(self):

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} Training Curve')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.save_dir, f'{self.model_name}_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to {save_path}")


# ===========================
# Main Function
# ===========================

def main():

    SAVE_DIR = "/content/drive/MyDrive/text2code-seq2seq/checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    config = {
        'num_train': 10000,
        'num_val': 1000,
        'num_test': 1000,
        'max_docstring_len': 50,
        'max_code_len': 80,
        'embedding_dim': 256,
        'hidden_dim': 256,
        'batch_size': 32,
        'num_epochs': 20,        # ← 20 থেকে 30
        'learning_rate': 0.0005, # ← 0.001 থেকে 0.0005
        'num_layers': 2,         # ← 1 থেকে 2 (MOST IMPORTANT!)
        'dropout': 0.3,          # ← Add dropout
        'resume': False
    }

    with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    models_to_train = [
        ('vanilla_rnn', create_vanilla_rnn_model),
        ('lstm', create_lstm_model),
        ('lstm_attention', create_lstm_attention_model)
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
            dropout=config.get('dropout', 0.0),  # ← এটা add করো
            device=device
        )

    trainer = Trainer(model, model_name, device, SAVE_DIR)
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate']
    )
       

    print("\nALL MODELS TRAINED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
