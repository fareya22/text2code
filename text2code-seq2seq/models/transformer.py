"""
Transformer Seq2Seq Model
Encoder-Decoder Transformer architecture for docstring-to-code generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=500, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder with embedding and positional encoding."""

    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=2, dim_feedforward=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch, src_len)
        embedded = self.pos_encoding(self.embedding(src) * self.scale)
        return self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )


class TransformerDecoder(nn.Module):
    """Transformer decoder with embedding, positional encoding, and output projection."""

    def __init__(self, vocab_size, d_model=256, nhead=8,
                 num_layers=2, dim_feedforward=512, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # trg: (batch, trg_len)
        embedded = self.pos_encoding(self.embedding(trg) * self.scale)
        output = self.transformer_decoder(
            embedded, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)


class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq model for docstring-to-code generation."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        """Generate causal mask: True means masked (ignored)."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=None):
        # teacher_forcing_ratio is accepted but unused (for API compat)
        src_key_padding_mask = (src == 0)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Shift target: decoder input is trg[:, :-1], predictions target trg[:, 1:]
        trg_input = trg[:, :-1]
        tgt_key_padding_mask = (trg_input == 0)
        tgt_mask = self._generate_square_subsequent_mask(
            trg_input.size(1), self.device
        )

        # (batch, trg_len-1, vocab_size)
        decoder_output = self.decoder(
            trg_input, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Pad with zeros at position 0 so output[:, 1:] gives the real predictions
        # This matches train_utils which does output[:, 1:] vs trg[:, 1:]
        batch_size = decoder_output.size(0)
        vocab_size = decoder_output.size(2)
        pad_col = torch.zeros(batch_size, 1, vocab_size, device=self.device)
        outputs = torch.cat([pad_col, decoder_output], dim=1)

        return outputs

    def generate(self, src, max_len, sos_idx, eos_idx):
        """Generate code autoregressively."""
        batch_size = src.shape[0]
        src_key_padding_mask = (src == 0)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Start with SOS token
        generated = torch.full(
            (batch_size, 1), sos_idx, dtype=torch.long, device=self.device
        )
        
        for _ in range(max_len - 1):
            tgt_key_padding_mask = (generated == 0)
            tgt_mask = self._generate_square_subsequent_mask(
                generated.size(1), self.device
            )
            
            decoder_output = self.decoder(
                generated, memory, tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Get the last token prediction
            next_token_logits = decoder_output[:, -1, :]
            next_token = next_token_logits.argmax(dim=1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generate EOS token
            if (next_token == eos_idx).all():
                break
        
        return generated


def create_transformer_model(input_vocab_size, output_vocab_size,
                            embedding_dim=256, hidden_dim=256,
                            num_layers=2, dropout=0.1, device='cpu'):
    """Factory function to create Transformer Seq2Seq model."""
    encoder = TransformerEncoder(
        input_vocab_size,
        d_model=embedding_dim,
        nhead=8,
        num_layers=num_layers,
        dim_feedforward=hidden_dim,
        dropout=dropout
    )
    
    decoder = TransformerDecoder(
        output_vocab_size,
        d_model=embedding_dim,
        nhead=8,
        num_layers=num_layers,
        dim_feedforward=hidden_dim,
        dropout=dropout
    )
    
    model = TransformerSeq2Seq(encoder, decoder, device).to(device)
    return model
