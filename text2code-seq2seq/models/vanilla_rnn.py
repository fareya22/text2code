"""
Vanilla RNN Seq2Seq Model
Basic encoder-decoder architecture without attention
"""

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    """Vanilla RNN Encoder"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.rnn(embedded, hidden)
        return outputs, hidden


class DecoderRNN(nn.Module):
    """Vanilla RNN Decoder"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)
        
    def forward(self, input_token, hidden):
        embedded = self.embedding(input_token)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class VanillaSeq2Seq(nn.Module):
    """Complete Vanilla RNN Seq2Seq Model"""
    
    def __init__(self, encoder, decoder, device):
        super(VanillaSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(2)
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs
    
    def generate(self, src, max_len, sos_idx, eos_idx):
        batch_size = src.shape[0]
        encoder_outputs, hidden = self.encoder(src)
        decoder_input = torch.LongTensor([[sos_idx]] * batch_size).to(self.device)
        
        generated = []
        for t in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden)
            top1 = output.argmax(2)
            generated.append(top1)
            decoder_input = top1
            
        generated = torch.cat(generated, dim=1)
        return generated


def create_vanilla_rnn_model(input_vocab_size, output_vocab_size, 
                             embedding_dim=128, hidden_dim=256, device='cpu', **kwargs):
    """Factory function to create Vanilla RNN Seq2Seq model"""
    encoder = EncoderRNN(input_vocab_size, embedding_dim, hidden_dim)
    decoder = DecoderRNN(output_vocab_size, embedding_dim, hidden_dim)
    model = VanillaSeq2Seq(encoder, decoder, device).to(device)
    return model