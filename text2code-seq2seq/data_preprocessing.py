"""
Data Preprocessing for CodeSearchNet Dataset
Handles loading, tokenization, and vocabulary building
"""

import torch
from datasets import load_dataset, DatasetDict, Dataset
from collections import Counter
import pickle
import os


class Vocabulary:
    """Vocabulary for text/code tokenization"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_count = Counter()
        self.n_words = 4  # Count SOS, EOS, PAD, UNK
        
    def add_sentence(self, sentence):
        """Add all words in a sentence to vocabulary"""
        for word in sentence.split():
            self.add_word(word)
            
    def add_word(self, word):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
        
    def __len__(self):
        return self.n_words


def simple_tokenize(text):
    """Simple whitespace tokenization"""
    # Basic cleaning
    text = text.strip().lower()
    # Split by whitespace
    tokens = text.split()
    return ' '.join(tokens)


def load_and_preprocess_data(num_train=10000, num_val=1000, num_test=1000, 
                             max_docstring_len=50, max_code_len=80):
    """
    Load CodeSearchNet dataset and preprocess
    
    Args:
        num_train: Number of training examples
        num_val: Number of validation examples
        num_test: Number of test examples
        max_docstring_len: Maximum docstring length
        max_code_len: Maximum code length
    
    Returns:
        train_data, val_data, test_data, docstring_vocab, code_vocab
    """
    
    print("Loading dataset...")
    dataset = load_dataset("Nan-Do/code-search-net-python")
    if 'validation' not in dataset:
        print("Validation split not found. Creating validation split from training data (10% of train set).")
        # Using a fixed test_size of 0.1 (10%) for creating the validation split.
        # This ensures a validation set is always present if `num_val` is intended as a cap
        # or if the original dataset simply lacks a 'validation' split.
        train_val_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = train_val_split['train']
        dataset['validation'] = train_val_split['test']

        if 'test' not in dataset:
            print("Test split not found. Creating test split from training data (10% of train set).")
            # Using a fixed test_size of 0.1 (10%) for creating the test split.
            # This ensures a test set is always present if the original dataset lacks a 'test' split.
            train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['train'] = train_test_split['train']
            dataset['test'] = train_test_split['test']

    # Select subsets
    train_raw = dataset['train'].select(range(min(num_train, len(dataset['train']))))
    val_raw = dataset['validation'].select(range(min(num_val, len(dataset['validation']))))
    test_raw = dataset['test'].select(range(min(num_test, len(dataset['test']))))
    
    print(f"Loaded {len(train_raw)} training, {len(val_raw)} validation, {len(test_raw)} test examples")
    
    # Build vocabularies
    print("Building vocabularies...")
    docstring_vocab = Vocabulary()
    code_vocab = Vocabulary()
    
    # Process data
    def process_split(data_split):
        processed = []
        for example in data_split:
            # Get docstring and code
            docstring = example.get('func_documentation_string', '') or example.get('docstring', '')
            code = example.get('func_code_string', '') or example.get('code', '')
            
            if not docstring or not code:
                continue
                
            # Tokenize
            docstring_tokens = simple_tokenize(docstring)
            code_tokens = simple_tokenize(code)
            
            # Check length constraints
            if len(docstring_tokens.split()) <= max_docstring_len and len(code_tokens.split()) <= max_code_len:
                processed.append({
                    'docstring': docstring_tokens,
                    'code': code_tokens
                })
                
        return processed
    
    train_data = process_split(train_raw)
    val_data = process_split(val_raw)
    test_data = process_split(test_raw)
    
    print(f"After filtering: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Build vocabularies from training data
    for example in train_data:
        docstring_vocab.add_sentence(example['docstring'])
        code_vocab.add_sentence(example['code'])
    
    print(f"Docstring vocabulary size: {len(docstring_vocab)}")
    print(f"Code vocabulary size: {len(code_vocab)}")
    
    return train_data, val_data, test_data, docstring_vocab, code_vocab


def sentence_to_indices(sentence, vocab, max_len):
    """Convert sentence to indices with padding"""
    indices = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) 
               for word in sentence.split()]
    
    # Truncate if too long
    indices = indices[:max_len]
    
    # Pad if too short
    while len(indices) < max_len:
        indices.append(vocab.word2idx['<PAD>'])
    
    return indices


def indices_to_sentence(indices, vocab):
    """Convert indices back to sentence"""
    words = []
    for idx in indices:
        if idx == vocab.word2idx['<EOS>']:
            break
        if idx != vocab.word2idx['<PAD>'] and idx != vocab.word2idx['<SOS>']:
            words.append(vocab.idx2word[idx])
    return ' '.join(words)


class CodeDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for code generation"""
    
    def __init__(self, data, docstring_vocab, code_vocab, max_docstring_len=50, max_code_len=80):
        self.data = data
        self.docstring_vocab = docstring_vocab
        self.code_vocab = code_vocab
        self.max_docstring_len = max_docstring_len
        self.max_code_len = max_code_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Convert to indices
        docstring_indices = sentence_to_indices(
            example['docstring'], 
            self.docstring_vocab, 
            self.max_docstring_len
        )
        
        # Add SOS and EOS to code
        code_words = example['code'].split()
        code_with_sos_eos = ['<SOS>'] + code_words + ['<EOS>']
        code_sentence = ' '.join(code_with_sos_eos)
        
        code_indices = sentence_to_indices(
            code_sentence,
            self.code_vocab,
            self.max_code_len + 2  # +2 for SOS and EOS
        )
        
        return {
            'docstring': torch.LongTensor(docstring_indices),
            'code': torch.LongTensor(code_indices),
            'docstring_text': example['docstring'],
            'code_text': example['code']
        }


def save_vocab(vocab, filepath):
    """Save vocabulary to file"""
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved to {filepath}")


def load_vocab(filepath):
    """Load vocabulary from file"""
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab


if __name__ == "__main__":
    # Test data preprocessing
    train_data, val_data, test_data, docstring_vocab, code_vocab = load_and_preprocess_data(
        num_train=1000, 
        num_val=100, 
        num_test=100
    )
    
    print("\nSample example:")
    print("Docstring:", train_data[0]['docstring'])
    print("Code:", train_data[0]['code'])
    
    # Test dataset
    dataset = CodeDataset(train_data[:10], docstring_vocab, code_vocab)
    sample = dataset[0]
    print("\nDataset sample shape:")
    print("Docstring tensor:", sample['docstring'].shape)
    print("Code tensor:", sample['code'].shape)