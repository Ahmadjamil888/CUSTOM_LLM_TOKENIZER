from sentence_transformers import SentenceTransformer
import numpy as np

# Load the pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change to other models

def generate_embeddings(tokenized_texts):
    """
    Generate vector embeddings for a list of tokenized text sequences.

    Args:
        tokenized_texts (List[str]): List of preprocessed and tokenized text strings.

    Returns:
        np.ndarray: Array of embeddings.
    """
    if not isinstance(tokenized_texts, list):
        raise ValueError("Input must be a list of tokenized strings.")
    
    # Join tokens to form sentences if passed as tokens
    texts = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in tokenized_texts]
    
    # Generate embeddings using SentenceTransformer
    embeddings = model.encode(texts)
    return np.array(embeddings)
