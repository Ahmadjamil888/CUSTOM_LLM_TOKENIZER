# scripts/generate_embeddings.py

import os
import json
from app.utils import read_file, clean_text, write_file
from app.tokenizer import tokenize_text, load_bpe_tokenizer, train_bpe_tokenizer
from app.embedding import generate_embeddings

DATA_PATH = "data/sample_input.txt"
OUTPUT_PATH = "processed/output.json"
TOKENIZER_PATH = "models/bpe_tokenizer.model"

def generate():
    # Step 1: Read and clean text
    print("[*] Reading and cleaning input text...")
    raw_text = read_file(DATA_PATH)
    cleaned_text = clean_text(raw_text)

    # Step 2: Ensure tokenizer is trained
    if not os.path.exists(TOKENIZER_PATH):
        print("[*] Tokenizer not found. Training tokenizer...")
        train_bpe_tokenizer([DATA_PATH])
    else:
        print("[*] Found existing tokenizer.")

    # Step 3: Tokenize sentences
    print("[*] Tokenizing text...")
    sentences = cleaned_text.split(".")
    tokenized_sentences = [tokenize_text(sentence) for sentence in sentences if sentence.strip()]

    # Step 4: Generate embeddings
    print("[*] Generating embeddings...")
    embeddings_json = generate_embeddings(tokenized_sentences)

    # Step 5: Save embeddings
    print(f"[*] Saving embeddings to {OUTPUT_PATH}...")
    write_file(OUTPUT_PATH, embeddings_json)

    print("[✓] Embedding generation complete.")

if __name__ == "__main__":
    generate()
