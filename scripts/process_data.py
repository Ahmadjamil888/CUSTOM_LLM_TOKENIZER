# scripts/process_data.py

import os
import json
from app.utils import read_file, clean_text, write_file
from app.tokenizer import tokenize_text, train_bpe_tokenizer, load_bpe_tokenizer
from app.embedding import generate_embeddings

DATA_PATH = "data/sample_input.txt"
OUTPUT_PATH = "processed/output.json"
TOKENIZER_PATH = "models/bpe_tokenizer.model"

def process():
    # 1. Read and clean the input data
    print("[*] Reading and cleaning input...")
    raw_text = read_file(DATA_PATH)
    cleaned_text = clean_text(raw_text)

    # 2. Train tokenizer if not already trained
    if not os.path.exists(TOKENIZER_PATH):
        print("[*] Tokenizer not found. Training new tokenizer...")
        train_bpe_tokenizer([DATA_PATH])
    else:
        print("[*] Using existing tokenizer.")

    # 3. Tokenize text into sentences and then words
    print("[*] Tokenizing text...")
    sentences = cleaned_text.split(".")  # crude sentence splitting
    tokenized_sentences = [tokenize_text(sentence) for sentence in sentences if sentence.strip()]

    # 4. Generate embeddings
    print("[*] Generating embeddings...")
    embeddings_json = generate_embeddings(tokenized_sentences)

    # 5. Save the output
    print(f"[*] Saving output to {OUTPUT_PATH}")
    write_file(OUTPUT_PATH, embeddings_json)
    print("[✓] Processing complete.")

if __name__ == "__main__":
    process()
