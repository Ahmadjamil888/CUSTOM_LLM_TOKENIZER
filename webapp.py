# webapp.py

from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
from app.utils import save_uploaded_file, clean_text, write_file
from app.tokenizer import tokenize_text, train_bpe_tokenizer
from app.embedding import generate_embeddings

UPLOAD_FOLDER = "data"
OUTPUT_FOLDER = "processed"
TOKENIZER_PATH = "models/bpe_tokenizer.model"
ALLOWED_EXTENSIONS = {"txt"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", output_file=None)


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = save_uploaded_file(file, app.config["UPLOAD_FOLDER"])
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Step 1: Read and clean
        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)

        # Step 2: Train tokenizer if not already
        if not os.path.exists(TOKENIZER_PATH):
            train_bpe_tokenizer([input_path])

        # Step 3: Tokenize sentences
        sentences = cleaned_text.split(".")
        tokenized_sentences = [tokenize_text(sentence) for sentence in sentences if sentence.strip()]

        # Step 4: Generate embeddings
        embeddings_json = generate_embeddings(tokenized_sentences)

        # Step 5: Save output
        output_filename = "output.json"
        output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
        write_file(output_path, embeddings_json)

        return render_template("index.html", output_file=output_filename)

    return "Invalid file type. Only .txt files allowed.", 400


@app.route("/processed/<filename>")
def download_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
