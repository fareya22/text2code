import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import pickle

# Special tokens
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

FREQ_THRESHOLD = 2  # Minimum frequency for a token to be in vocab


def tokenize(text):
    """Regex-based tokenizer for code and docstrings"""
    text = text.strip()
    text = text.replace('\n', ' NEWLINE ').replace('\t', ' INDENT ')
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s]', text)
    return tokens


def simple_tokenize(text):
    """Simple tokenizer: lowercase and split by whitespace"""
    text = text.strip().lower()
    tokens = text.split()
    return " ".join(tokens)


def sentence_to_indices(sentence, vocab, max_len=10):
    """Convert a sentence to padded indices"""
    tokens = sentence.lower().split()
    indices = [vocab.word2idx.get(token, UNK_IDX) for token in tokens]
    # Pad to max_len
    if len(indices) < max_len:
        indices.extend([PAD_IDX] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]
    return indices


class Vocabulary:
    def __init__(self, freq_threshold=FREQ_THRESHOLD):
        self.freq_threshold = freq_threshold
        self.itos = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.counter = Counter()

    @property
    def word2idx(self):
        """Alias for stoi (string to index)"""
        return self.stoi

    def add_sentence(self, sentence):
        """Add tokens from a sentence to the vocabulary"""
        tokens = simple_tokenize(sentence).split()
        idx = len(self.stoi)
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def build_vocabulary(self, token_lists):
        for tokens in token_lists:
            self.counter.update(tokens)
        idx = len(self.itos)
        for token, count in self.counter.most_common():
            if count >= self.freq_threshold:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1

    def numericalize(self, tokens):
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, indices):
        tokens = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx not in (PAD_IDX, SOS_IDX):
                tokens.append(self.itos.get(idx, UNK_TOKEN))
        return tokens

    def __len__(self):
        return len(self.itos)


class CodeDocstringDataset(Dataset):
    def __init__(self, docstrings, codes, src_vocab, trg_vocab, max_src_len=50, max_trg_len=80):
        self.docstrings = docstrings
        self.codes = codes
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.docstrings)

    def __getitem__(self, idx):
        src_tokens = self.docstrings[idx][:self.max_src_len]
        trg_tokens = self.codes[idx][:self.max_trg_len]

        src_indices = [SOS_IDX] + self.src_vocab.numericalize(src_tokens) + [EOS_IDX]
        trg_indices = [SOS_IDX] + self.trg_vocab.numericalize(trg_tokens) + [EOS_IDX]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)


def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded


def load_and_prepare_data(dataset_name="code_search_net", num_train=10000, num_val=1000, num_test=1000,
                          max_src_len=50, max_trg_len=80, batch_size=32, freq_threshold=FREQ_THRESHOLD):
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("Nan-Do/code-search-net-python")

    # Create validation/test splits if missing
    if 'validation' not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = split['train']
        dataset['validation'] = split['test']

    if 'test' not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = split['train']
        dataset['test'] = split['test']

    # Select subsets
    train_raw = dataset['train'].select(range(min(num_train, len(dataset['train']))))
    val_raw = dataset['validation'].select(range(min(num_val, len(dataset['validation']))))
    test_raw = dataset['test'].select(range(min(num_test, len(dataset['test']))))

    # Tokenize
    def process_split(split_data):
        docstrings, codes = [], []
        for example in split_data:
            docstring = example.get('func_documentation_string', '') or example.get('docstring', '')
            code = example.get('func_code_string', '') or example.get('code', '')
            if not docstring or not code:
                continue
            doc_tokens = tokenize(docstring)
            code_tokens = tokenize(code)
            docstrings.append(doc_tokens)
            codes.append(code_tokens)
        return docstrings, codes

    train_docs, train_codes = process_split(train_raw)
    val_docs, val_codes = process_split(val_raw)
    test_docs, test_codes = process_split(test_raw)

    print(f"Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}")

    # Build vocabularies
    src_vocab = Vocabulary(freq_threshold=freq_threshold)
    src_vocab.build_vocabulary(train_docs)
    trg_vocab = Vocabulary(freq_threshold=freq_threshold)
    trg_vocab.build_vocabulary(train_codes)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")

    # Create datasets
    train_dataset = CodeDocstringDataset(train_docs, train_codes, src_vocab, trg_vocab, max_src_len, max_trg_len)
    val_dataset = CodeDocstringDataset(val_docs, val_codes, src_vocab, trg_vocab, max_src_len, max_trg_len)
    test_dataset = CodeDocstringDataset(test_docs, test_codes, src_vocab, trg_vocab, max_src_len, max_trg_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, trg_vocab


# Save/load vocab for future reuse
def save_vocab(vocab, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ===== WRAPPER FUNCTIONS FOR COMPATIBILITY =====

def load_and_preprocess_data(dataset_name="code_search_net", num_train=10000, num_val=1000, num_test=1000,
                             max_docstring_len=50, max_code_len=80, batch_size=32, freq_threshold=FREQ_THRESHOLD):
    """
    Wrapper function for compatibility with train.py/evaluate.py
    Returns preprocessed data (not dataloaders)
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("Nan-Do/code-search-net-python")

    # Create validation/test splits if missing
    if 'validation' not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = split['train']
        dataset['validation'] = split['test']

    if 'test' not in dataset:
        split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        dataset['train'] = split['train']
        dataset['test'] = split['test']

    # Select subsets
    train_raw = dataset['train'].select(range(min(num_train, len(dataset['train']))))
    val_raw = dataset['validation'].select(range(min(num_val, len(dataset['validation']))))
    test_raw = dataset['test'].select(range(min(num_test, len(dataset['test']))))

    # Tokenize
    def process_split(split_data):
        docstrings, codes = [], []
        for example in split_data:
            docstring = example.get('func_documentation_string', '') or example.get('docstring', '')
            code = example.get('func_code_string', '') or example.get('code', '')
            if not docstring or not code:
                continue
            doc_tokens = tokenize(docstring)
            code_tokens = tokenize(code)
            docstrings.append(doc_tokens)
            codes.append(code_tokens)
        return docstrings, codes

    train_docs, train_codes = process_split(train_raw)
    val_docs, val_codes = process_split(val_raw)
    test_docs, test_codes = process_split(test_raw)

    print(f"Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}")

    # Build vocabularies
    src_vocab = Vocabulary(freq_threshold=freq_threshold)
    src_vocab.build_vocabulary(train_docs)
    trg_vocab = Vocabulary(freq_threshold=freq_threshold)
    trg_vocab.build_vocabulary(train_codes)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")

    # Prepare data in format expected by train.py
    train_data = [(d, c) for d, c in zip(train_docs, train_codes)]
    val_data = [(d, c) for d, c in zip(val_docs, val_codes)]
    test_data = [(d, c) for d, c in zip(test_docs, test_codes)]

    return train_data, val_data, test_data, src_vocab, trg_vocab


class CodeDataset(Dataset):
    """Wrapper dataset class for compatibility with train.py"""
    def __init__(self, data, src_vocab, trg_vocab, max_src_len=50, max_trg_len=80):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_tokens, trg_tokens = self.data[idx]
        
        src_tokens = src_tokens[:self.max_src_len]
        trg_tokens = trg_tokens[:self.max_trg_len]

        src_indices = [SOS_IDX] + self.src_vocab.numericalize(src_tokens) + [EOS_IDX]
        trg_indices = [SOS_IDX] + self.trg_vocab.numericalize(trg_tokens) + [EOS_IDX]

        return {
            'docstring': torch.tensor(src_indices, dtype=torch.long),
            'code': torch.tensor(trg_indices, dtype=torch.long)
        }


def collate_batch(batch):
    """Custom collate function to pad sequences"""
    docstrings = [item['docstring'] for item in batch]
    codes = [item['code'] for item in batch]
    
    docstrings_padded = pad_sequence(docstrings, batch_first=True, padding_value=PAD_IDX)
    codes_padded = pad_sequence(codes, batch_first=True, padding_value=PAD_IDX)
    
    return {
        'docstring': docstrings_padded,
        'code': codes_padded
    }



def indices_to_sentence(indices, vocab):
    """Convert indices back to sentence string"""
    tokens = []
    for idx in indices:
        if idx == EOS_IDX or idx == PAD_IDX:
            continue
        if idx == SOS_IDX:
            continue
        tokens.append(vocab.itos.get(idx, '<UNK>'))
    return ' '.join(tokens)

