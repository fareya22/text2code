from data_preprocessing import Vocabulary, simple_tokenize, sentence_to_indices

# Test vocabulary
vocab = Vocabulary()
vocab.add_sentence("hello world this is a test")

print(f'Vocab size: {len(vocab)}')
print(f'stoi: {vocab.stoi}')
print(f'word2idx hello: {vocab.word2idx.get("hello")}')

# Test assertions
assert len(vocab) == 10, f"Expected 10, got {len(vocab)}"  # 4 special tokens + 6 words
assert vocab.word2idx['hello'] == 4, f"Expected 4, got {vocab.word2idx['hello']}"

# Test tokenization
text = "  Hello   World  "
tokens = simple_tokenize(text)
print(f'simple_tokenize result: "{tokens}"')
assert tokens == "hello world", f"Expected 'hello world', got '{tokens}'"

# Test indices conversion
indices = sentence_to_indices("hello world", vocab, max_len=10)
print(f'indices: {indices}')
print(f'vocab.word2idx["hello"]: {vocab.word2idx["hello"]}')
print(f'vocab.word2idx["world"]: {vocab.word2idx["world"]}')
assert len(indices) == 10
assert indices[0] == vocab.word2idx['hello']
assert indices[1] == vocab.word2idx['world']

print("All tests passed!")
