"""
Quick Test Script
Verify that all models can be instantiated and run forward pass
"""

import torch
import sys

def test_vanilla_rnn():
    """Test Vanilla RNN model"""
    print("\n" + "="*60)
    print("Testing Vanilla RNN Seq2Seq")
    print("="*60)
    
    from models.vanilla_rnn import create_vanilla_rnn_model
    
    device = torch.device('cpu')
    model = create_vanilla_rnn_model(
        input_vocab_size=1000,
        output_vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        device=device
    )
    
    # Test forward pass
    src = torch.randint(0, 1000, (4, 50))
    trg = torch.randint(0, 1000, (4, 80))
    
    output = model(src, trg, teacher_forcing_ratio=1.0)
    assert output.shape == (4, 80, 1000), f"Expected (4, 80, 1000), got {output.shape}"
    
    # Test generation
    generated = model.generate(src, max_len=80, sos_idx=1, eos_idx=2)
    assert generated.shape == (4, 80), f"Expected (4, 80), got {generated.shape}"
    
    print("✓ Vanilla RNN: All tests passed!")
    return True


def test_lstm():
    """Test LSTM model"""
    print("\n" + "="*60)
    print("Testing LSTM Seq2Seq")
    print("="*60)
    
    from models.lstm_seq2seq import create_lstm_model
    
    device = torch.device('cpu')
    model = create_lstm_model(
        input_vocab_size=1000,
        output_vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        device=device
    )
    
    # Test forward pass
    src = torch.randint(0, 1000, (4, 50))
    trg = torch.randint(0, 1000, (4, 80))
    
    output = model(src, trg, teacher_forcing_ratio=1.0)
    assert output.shape == (4, 80, 1000), f"Expected (4, 80, 1000), got {output.shape}"
    
    # Test generation
    generated = model.generate(src, max_len=80, sos_idx=1, eos_idx=2)
    assert generated.shape == (4, 80), f"Expected (4, 80), got {generated.shape}"
    
    print("✓ LSTM: All tests passed!")
    return True


def test_lstm_attention():
    """Test LSTM + Attention model"""
    print("\n" + "="*60)
    print("Testing LSTM + Attention Seq2Seq")
    print("="*60)
    
    from models.lstm_attention import create_lstm_attention_model
    
    device = torch.device('cpu')
    model = create_lstm_attention_model(
        input_vocab_size=1000,
        output_vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        device=device
    )
    
    # Test forward pass
    src = torch.randint(0, 1000, (4, 50))
    trg = torch.randint(0, 1000, (4, 80))
    
    output, attentions = model(src, trg, teacher_forcing_ratio=1.0)
    assert output.shape == (4, 80, 1000), f"Expected (4, 80, 1000), got {output.shape}"
    assert len(attentions) == 79, f"Expected 79 attention weights, got {len(attentions)}"
    assert attentions[0].shape == (4, 50), f"Expected (4, 50), got {attentions[0].shape}"
    
    # Test generation
    generated, gen_attentions = model.generate(src, max_len=80, sos_idx=1, eos_idx=2)
    assert generated.shape == (4, 80), f"Expected (4, 80), got {generated.shape}"
    assert len(gen_attentions) == 80, f"Expected 80 attention weights, got {len(gen_attentions)}"
    
    print("✓ LSTM + Attention: All tests passed!")
    return True


def test_transformer():
    """Test Transformer model"""
    print("\n" + "="*60)
    print("Testing Transformer Seq2Seq")
    print("="*60)
    
    from models.transformer import create_transformer_model
    
    device = torch.device('cpu')
    model = create_transformer_model(
        input_vocab_size=1000,
        output_vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1,
        dropout=0.1,
        device=device
    )
    
    # Test forward pass
    src = torch.randint(0, 1000, (4, 50))
    trg = torch.randint(0, 1000, (4, 80))
    
    output = model(src, trg, teacher_forcing_ratio=None)
    assert output.shape == (4, 80, 1000), f"Expected (4, 80, 1000), got {output.shape}"
    
    # Test generation
    generated = model.generate(src, max_len=80, sos_idx=1, eos_idx=2)
    assert generated.shape[0] == 4, f"Expected batch size 4, got {generated.shape[0]}"
    assert generated.shape[1] <= 80, f"Expected sequence length <= 80, got {generated.shape[1]}"
    
    print("✓ Transformer: All tests passed!")
    return True


def test_data_preprocessing():
    """Test data preprocessing"""
    print("\n" + "="*60)
    print("Testing Data Preprocessing")
    print("="*60)
    
    from data_preprocessing import Vocabulary, simple_tokenize, sentence_to_indices
    
    # Test vocabulary
    vocab = Vocabulary()
    vocab.add_sentence("hello world this is a test")
    
    assert len(vocab) == 10  # 4 special tokens + 6 words
    assert vocab.word2idx['hello'] == 4
    
    # Test tokenization
    text = "  Hello   World  "
    tokens = simple_tokenize(text)
    assert tokens == "hello world"
    
    # Test indices conversion
    indices = sentence_to_indices("hello world", vocab, max_len=10)
    assert len(indices) == 10
    assert indices[0] == vocab.word2idx['hello']
    assert indices[1] == vocab.word2idx['world']
    
    print("✓ Data Preprocessing: All tests passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING MODEL TESTS")
    print("="*60)
    
    all_passed = True
    
    try:
        test_data_preprocessing()
    except Exception as e:
        print(f"✗ Data Preprocessing failed: {e}")
        all_passed = False
    
    try:
        test_vanilla_rnn()
    except Exception as e:
        print(f"✗ Vanilla RNN failed: {e}")
        all_passed = False
    
    try:
        test_lstm()
    except Exception as e:
        print(f"✗ LSTM failed: {e}")
        all_passed = False
    
    try:
        test_lstm_attention()
    except Exception as e:
        print(f"✗ LSTM + Attention failed: {e}")
        all_passed = False
    
    try:
        test_transformer()
    except Exception as e:
        print(f"✗ Transformer failed: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou're ready to train! Run:")
        print("  python train.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the errors above before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
