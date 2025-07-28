# main.py

import os
import argparse
from app.utils import read_file, clean_text, write_file
from app.tokenizer import tokenize_text, train_bpe_tokenizer
from app.embedding import generate_embeddings

DEFAULT_INPUT = "data/sample_input.txt"
DEFAULT_OUTPUT = "processed/output.json"
TOKENIZER_PATH = "models/bpe_tokenizer.model"

def run_pipeline(input_file, output_file):
    print(f"[*] Processing: {input_file}")

    # 1. Read and clean
    print("[*] Reading and cleaning input...")
    raw_text = read_file(input_file)
    cleaned = clean_text(raw_text)

    # 2. Train tokenizer if not exists
    if not os.path.exists(TOKENIZER_PATH):
        print("[*] No tokenizer found, training new tokenizer...")
        train_bpe_tokenizer([input_file])
    else:
        print("[*] Using existing tokenizer.")

    # 3. Tokenize
    print("[*] Tokenizing text...")
    sentences = cleaned.split(".")
    tokenized = [tokenize_text(s) for s in sentences if s.strip()]

    # 4. Generate embeddings
    print("[*] Generating embeddings...")
    embeddings = generate_embeddings(tokenized)

    # 5. Save output
    print(f"[*] Writing output to: {output_file}")
    write_file(output_file, embeddings)
    print("[✓] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BPE tokenizer + embedding generator pipeline.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Path to input text file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Path to save processed output")
    
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
