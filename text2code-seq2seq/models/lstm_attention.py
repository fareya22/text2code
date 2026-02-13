"""
LSTM with Attention Mechanism
Bahdanau (additive) attention with dropout support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalEncoderLSTM(nn.Module):
    """Bidirectional LSTM Encoder with Dropout"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(BidirectionalEncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        
        # Dropout only works with num_layers > 1
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Combine bidirectional states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_combined = torch.tanh(self.fc_hidden(hidden_combined))
        hidden_combined = hidden_combined.unsqueeze(0)
        
        cell_fwd = cell[-2, :, :]
        cell_bwd = cell[-1, :, :]
        cell_combined = torch.cat([cell_fwd, cell_bwd], dim=1)
        cell_combined = torch.tanh(self.fc_cell(cell_combined))
        cell_combined = cell_combined.unsqueeze(0)
        
        return outputs, (hidden_combined, cell_combined)


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention Mechanism"""
    
    def __init__(self, encoder_dim, decoder_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.W1 = nn.Linear(encoder_dim, decoder_dim)
        self.W2 = nn.Linear(decoder_dim, decoder_dim)
        self.V = nn.Linear(decoder_dim, 1)
        
    def forward(self, encoder_outputs, decoder_hidden):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
        attention_scores = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)
        
        return attention_weights, context_vector


class AttentionDecoderLSTM(nn.Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim, encoder_dim):
        super(AttentionDecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input_token, hidden, cell, encoder_outputs):
        embedded = self.embedding(input_token)
        
        decoder_hidden = hidden.squeeze(0)
        attention_weights, context_vector = self.attention(encoder_outputs, decoder_hidden)
        
        context_vector = context_vector.unsqueeze(1)
        lstm_input = torch.cat([embedded, context_vector], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.fc(output)
        
        return output, (hidden, cell), attention_weights


class LSTMAttentionSeq2Seq(nn.Module):
    """Complete LSTM Seq2Seq with Attention"""
    
    def __init__(self, encoder, decoder, device):
        super(LSTMAttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = []
        
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, (hidden, cell), attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output.squeeze(1)
            attentions.append(attention_weights)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs, attentions
    
    def generate(self, src, max_len, sos_idx, eos_idx):
        batch_size = src.shape[0]
        encoder_outputs, (hidden, cell) = self.encoder(src)
        decoder_input = torch.LongTensor([[sos_idx]] * batch_size).to(self.device)
        
        generated = []
        attentions = []
        
        for t in range(max_len):
            output, (hidden, cell), attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            top1 = output.argmax(2)
            generated.append(top1)
            attentions.append(attention_weights)
            decoder_input = top1
            
        generated = torch.cat(generated, dim=1)
        return generated, attentions


def create_lstm_attention_model(input_vocab_size, output_vocab_size, 
                                embedding_dim=128, hidden_dim=256, 
                                num_layers=1, dropout=0.0, device='cpu'):
    """Factory function to create LSTM Seq2Seq with Attention and dropout"""
    encoder = BidirectionalEncoderLSTM(
        input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout
    )
    
    decoder = AttentionDecoderLSTM(
        output_vocab_size, embedding_dim, hidden_dim, encoder_dim=hidden_dim * 2
    )
    
    model = LSTMAttentionSeq2Seq(encoder, decoder, device).to(device)
    return model