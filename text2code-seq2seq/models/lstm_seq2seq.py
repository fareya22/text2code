"""
LSTM Seq2Seq Model
Improved encoder-decoder with LSTM cells and dropout support
"""

import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    """LSTM Encoder with Dropout"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        
        # Dropout only works with num_layers > 1
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded, hidden)
        return outputs, (hidden, cell)


class DecoderLSTM(nn.Module):
    """LSTM Decoder with Dropout"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        
        # Dropout only works with num_layers > 1
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input_token, hidden, cell):
        embedded = self.embedding(input_token)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output)
        return output, (hidden, cell)


class LSTMSeq2Seq(nn.Module):
    """Complete LSTM Seq2Seq Model"""
    
    def __init__(self, encoder, decoder, device):
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs
    
    def generate(self, src, max_len, sos_idx, eos_idx):
        batch_size = src.shape[0]
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = torch.LongTensor([[sos_idx]] * batch_size).to(self.device)
        
        generated = []
        for t in range(max_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)
            top1 = output.argmax(2)
            generated.append(top1)
            decoder_input = top1
            
        generated = torch.cat(generated, dim=1)
        return generated


def create_lstm_model(input_vocab_size, output_vocab_size, 
                      embedding_dim=128, hidden_dim=256, 
                      num_layers=1, dropout=0.0, device='cpu'):
    """Factory function to create LSTM Seq2Seq model with dropout"""
    encoder = EncoderLSTM(input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    decoder = DecoderLSTM(output_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    model = LSTMSeq2Seq(encoder, decoder, device).to(device)
    return model