# scripts/train_word2vec.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# Add project root to path before importing local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils import clean_text

# Hyperparameters
EMBEDDING_DIM = 100
WINDOW_SIZE = 2
EPOCHS = 5
MIN_COUNT = 2
LEARNING_RATE = 0.01

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, all_context_words):
        center_vectors = self.input_embeddings(center_words)
        context_vectors = self.output_embeddings(all_context_words)
        scores = torch.matmul(center_vectors, context_vectors.T)
        return scores

def load_text_corpus(data_dir):
    corpus = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                for line in file:
                    cleaned = clean_text(line)
                    if cleaned:
                        corpus.extend(cleaned.strip().split())
    return corpus

def build_vocab(corpus):
    word_counts = Counter(corpus)
    vocab = {word for word, count in word_counts.items() if count >= MIN_COUNT}
    word_to_ix = {word: idx for idx, word in enumerate(vocab)}
    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    return word_to_ix, ix_to_word

def generate_training_pairs(corpus, word_to_ix):
    pairs = []
    for idx, word in enumerate(corpus):
        if word not in word_to_ix:
            continue
        start = max(0, idx - WINDOW_SIZE)
        end = min(len(corpus), idx + WINDOW_SIZE + 1)
        for j in range(start, end):
            if idx != j and corpus[j] in word_to_ix:
                pairs.append((word_to_ix[word], word_to_ix[corpus[j]]))
    return pairs

def train_word2vec():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    model_save_path = os.path.join(os.path.dirname(__file__), "..", "models", "word2vec.pt")

    print("[INFO] Loading and preprocessing data...")
    corpus = load_text_corpus(data_dir)

    print("[INFO] Building vocabulary...")
    word_to_ix, ix_to_word = build_vocab(corpus)
    if not word_to_ix:
        raise ValueError("Empty vocabulary. Consider reducing MIN_COUNT.")

    print("[INFO] Generating training pairs...")
    training_pairs = generate_training_pairs(corpus, word_to_ix)

    print(f"[INFO] Training Skip-Gram model on {len(training_pairs)} pairs with vocab size {len(word_to_ix)}")
    model = SkipGramModel(len(word_to_ix), EMBEDDING_DIM)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for center, context in training_pairs:
            center_tensor = torch.tensor([center], dtype=torch.long)
            context_tensor = torch.tensor([context], dtype=torch.long)

            model.zero_grad()
            all_contexts = torch.arange(len(word_to_ix))
            predictions = model(center_tensor, all_contexts)
            loss = loss_function(predictions, context_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss:.4f}")

    print("[INFO] Saving model and vocab...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word_to_ix": word_to_ix,
        "ix_to_word": ix_to_word
    }, model_save_path)
    print(f"[SUCCESS] Model saved to {model_save_path}")

if __name__ == "__main__":
    train_word2vec()
