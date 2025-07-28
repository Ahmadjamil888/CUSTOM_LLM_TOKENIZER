# scripts/train_tokenizer.py

import os
import sys

# Add root directory to Python path so app module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.tokenizer import train_bpe_tokenizer

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not data_files:
        raise FileNotFoundError("No training files found in the 'data/' folder.")

    print(f"[INFO] Training tokenizer on files:\n{data_files}")
    tokenizer = train_bpe_tokenizer(data_files)
    print("[SUCCESS] Tokenizer trained and saved to models/bpe_tokenizer.model")

if __name__ == "__main__":
    main()
