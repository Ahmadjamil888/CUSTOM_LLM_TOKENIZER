# app/utils.py

import os
import re
import json
import numpy as np
from werkzeug.utils import secure_filename

def clean_text(text):
    """
    Clean input text by removing unwanted characters, extra spaces, etc.
    Args:
        text: Raw text string
    Returns:
        Cleaned text
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,!?;:()'\"]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def save_uploaded_file(file, upload_folder):
    """
    Save uploaded file to disk.
    Args:
        file: werkzeug FileStorage object
        upload_folder: path to save file
    Returns:
        filename: the name of the saved file
    """
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = secure_filename(file.filename)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    return filename

def read_file(filepath):
    """
    Read text from a file.
    Args:
        filepath: path to input file
    Returns:
        content: string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    """
    Write content to a file. Supports strings, dicts, lists, and NumPy arrays.
    Args:
        filepath: path to output file
        content: string, dict, list, or numpy.ndarray
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        if isinstance(content, np.ndarray):
            json.dump(content.tolist(), f)
        elif isinstance(content, (dict, list)):
            json.dump(content, f)
        else:
            f.write(str(content))
