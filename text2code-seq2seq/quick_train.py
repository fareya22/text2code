"""
Quick Training Script - Minimal Dataset & Epochs
Generate vocab pkl files quickly for evaluation testing
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from data_preprocessing import (
    load_and_preprocess_data,
    CodeDataset,
    save_vocab,
    load_vocab,
    collate_batch
)

from models.vanilla_rnn import create_vanilla_rnn_model
from models.lstm_seq2seq import create_lstm_model
from models.lstm_attention import create_lstm_attention_model


class QuickTrainer:
    def __init__(self, model, model_name, device, save_dir):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5):
        self.model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Training {self.model_name}", leave=False):
            src = batch['docstring'].to(self.device)
            trg = batch['code'].to(self.device)
            optimizer.zero_grad()
            
            if 'attention' in self.model_name.lower():
                output, _ = self.model(src, trg, teacher_forcing_ratio)
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
        
        return epoch_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validating {self.model_name}", leave=False):
                src = batch['docstring'].to(self.device)
                trg = batch['code'].to(self.device)
                
                if 'attention' in self.model_name.lower():
                    output, _ = self.model(src, trg, 0)
                else:
                    output = self.model(src, trg, 0)
                
                output_dim = output.shape[-1]
                output = output[:,1:].reshape(-1, output_dim)
                trg = trg[:,1:].reshape(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        
        return epoch_loss / len(dataloader)

    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is PAD_IDX
        
        best_val_loss = float('inf')
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} (QUICK MODE - {num_epochs} epochs)")
        print(f"{'='*60}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_losses': [train_loss],
                    'val_losses': [val_loss]
                }
                torch.save(checkpoint, os.path.join(self.save_dir, f'{self.model_name}_best.pt'))
            
            # Always save latest
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_losses': [train_loss],
                'val_losses': [val_loss]
            }
            torch.save(checkpoint, os.path.join(self.save_dir, f'{self.model_name}_latest.pt'))
        
        print(f"✓ {self.model_name} training complete! Best Val Loss: {best_val_loss:.4f}\n")


def main():
    SAVE_DIR = 'checkpoints'
    
    # QUICK CONFIG - সময় কম লাগবে
    config = {
        "num_train": 1000,          # ← Reduce from 10000
        "num_val": 300,             # ← Reduce from 1500
        "num_test": 300,            # ← Reduce from 1500
        "max_docstring_len": 50,
        "max_code_len": 80,
        "embedding_dim": 128,       # ← Reduce from 256
        "hidden_dim": 128,          # ← Reduce from 256
        "batch_size": 32,           # ← Reduce from 64
        "num_epochs": 2,            # ← Reduce from 15
        "learning_rate": 0.001,
        "num_layers": 1,            # ← Reduce from 2
        "dropout": 0.3
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n⚡ QUICK TRAINING MODE: {config['num_train']} samples, {config['num_epochs']} epochs")
    print(f"Estimated time: 5-10 minutes (GPU) or 30-60 minutes (CPU)\n")

    print("Loading and preprocessing data...")
    train_data, val_data, test_data, docstring_vocab, code_vocab = load_and_preprocess_data(
        num_train=config['num_train'],
        num_val=config['num_val'],
        num_test=config['num_test'],
        max_docstring_len=config['max_docstring_len'],
        max_code_len=config['max_code_len']
    )

    print("Saving vocabularies...")
    save_vocab(docstring_vocab, os.path.join(SAVE_DIR, 'docstring_vocab.pkl'))
    save_vocab(code_vocab, os.path.join(SAVE_DIR, 'code_vocab.pkl'))

    print(f"Docstring vocab size: {len(docstring_vocab)}")
    print(f"Code vocab size: {len(code_vocab)}\n")

    train_dataset = CodeDataset(train_data, docstring_vocab, code_vocab,
                                config['max_docstring_len'], config['max_code_len'])
    val_dataset = CodeDataset(val_data, docstring_vocab, code_vocab,
                              config['max_docstring_len'], config['max_code_len'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_batch)

    models_to_train = [
        ('vanilla_rnn', create_vanilla_rnn_model),
        ('lstm', create_lstm_model),
        ('lstm_attention', create_lstm_attention_model)
    ]

    for model_name, model_factory in models_to_train:
        print(f"Creating {model_name} model...")
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
        
        trainer = QuickTrainer(model, model_name, device, SAVE_DIR)
        trainer.train(
            train_loader, 
            val_loader, 
            num_epochs=config['num_epochs'], 
            learning_rate=config['learning_rate']
        )

    print("="*60)
    print("✓ QUICK TRAINING COMPLETE!")
    print("="*60)
    print("\n✅ Pkl files created:")
    print("   - checkpoints/docstring_vocab.pkl")
    print("   - checkpoints/code_vocab.pkl")
    print("\n✅ Models saved:")
    print("   - checkpoints/vanilla_rnn_best.pt & latest.pt")
    print("   - checkpoints/lstm_best.pt & latest.pt")
    print("   - checkpoints/lstm_attention_best.pt & latest.pt")
    print("\n🚀 Now run: python evaluate.py")


if __name__ == "__main__":
    main()
