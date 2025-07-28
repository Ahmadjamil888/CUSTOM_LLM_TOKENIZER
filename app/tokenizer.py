# app/tokenizer.py

import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing

# Path where the trained tokenizer will be saved
TOKENIZER_PATH = "models/bpe_tokenizer.model"

def train_bpe_tokenizer(data_files, vocab_size=10000):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the provided text files.

    Parameters:
        data_files (list): Paths to training text files.
        vocab_size (int): Target vocabulary size.

    Returns:
        Tokenizer: Trained BPE tokenizer object.
    """
    tokenizer = Tokenizer(models.BPE())

    # Normalize and pre-tokenize text
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase()
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Define special tokens and trainer
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens)

    # Train tokenizer
    tokenizer.train(files=data_files, trainer=trainer)

    # Add post-processing template
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]"))
        ]
    )

    # Save the tokenizer
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)

    return tokenizer

def load_bpe_tokenizer():
    """
    Load a trained BPE tokenizer from disk.

    Returns:
        Tokenizer: Loaded BPE tokenizer.

    Raises:
        FileNotFoundError: If the tokenizer model file is missing.
    """
    if os.path.exists(TOKENIZER_PATH):
        return Tokenizer.from_file(TOKENIZER_PATH)
    else:
        raise FileNotFoundError("Tokenizer model not found. Train it using `train_bpe_tokenizer()` first.")

def tokenize_text(text):
    """
    Tokenize input text using the trained BPE tokenizer.

    Parameters:
        text (str): Raw input text.

    Returns:
        list: List of tokens.
    """
    tokenizer = load_bpe_tokenizer()
    encoded = tokenizer.encode(text)
    return encoded.tokens
