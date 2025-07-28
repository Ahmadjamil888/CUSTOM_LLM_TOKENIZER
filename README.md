# Custom NLP Tokenizer and Embedding Web App

This repository contains a complete pipeline for building a custom NLP tokenizer using the Byte-Pair Encoding (BPE) algorithm, training word embeddings from scratch (without Gensim), and providing a user-friendly web interface to upload, tokenize, and embed text. This project is ideal for NLP researchers, developers, and students looking to understand and implement tokenization and word embeddings using Python from the ground up.

##  Features

- Byte-Pair Encoding (BPE) Tokenizer training on raw `.txt` files.
- Word2Vec embedding model training without using external libraries like Gensim.
- Flask-based web app for uploading text files and generating embeddings.
- Modular and clean Python code with production-ready structure.
- SEO-optimized structure for educational and research use.

##  Project Structure

```
project/
│
├── app/
│   ├── __init__.py
│   ├── tokenizer.py         # BPE tokenizer logic
│   ├── embedding.py         # Word2Vec training logic
│   ├── utils.py             # Utilities for cleaning, saving, and reading files
│   └── webapp.py            # Flask web application
│
├── models/                  # Stores trained tokenizer and embedding models
│
├── data/                    # Training data (.txt files)
│
├── scripts/
│   ├── train_tokenizer.py   # Script to train BPE tokenizer
│   ├── train_embedding.py   # Script to train Word2Vec embedding model
│
├── README.md
```

##  Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ahmadjamil888/CUSTOM_LLM_TOKENIZER.git
cd CUSTOM_LLM_TOKENIZER
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train BPE Tokenizer

```bash
python scripts/train_tokenizer.py --input data/training_text.txt --vocab_size 1000 --output models/tokenizer.json
```

### 5. Train Word Embedding Model

```bash
python scripts/train_embedding.py --input data/training_text.txt --output models/word2vec_model.json
```

### 6. Run the Web App

```bash
python app/webapp.py
```

Visit `http://127.0.0.1:5000` in your browser to access the interface.

##  File Upload and Testing

1. Navigate to the web app URL.
2. Upload a `.txt` file.
3. The app will tokenize and embed the text using the trained models.
4. Results will be displayed on the page and optionally saved to disk.

##  Notes

- This app is for educational and research purposes.
- For production deployment, consider using `gunicorn` and `nginx`.
- Extend the app by adding visualization tools like t-SNE for embeddings.

##  License

This project is licensed under the MIT License.

---

For more details or collaboration inquiries, feel free to contact the repository maintainer.
