from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np

def generate_specter_embeddings(texts, model_name='allenai/specter2_base', batch_size=32):
    """
    Genera embeddings SPECTER2 para una lista de textos.
    Utiliza GPU si está disponible.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   [Embeddings] Usando dispositivo: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    
    # En SPECTER2 base, el modelo ya maneja el formato si le pasamos los strings
    # Pero para mejores resultados, el paper sugiere: "Title: [T] [SEP] Abstract: [A]"
    # Aquí asumimos que texts ya vienen pre-formateados o son solo el contenido.
    
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    return embeddings

def prepare_text_for_specter(df):
    """
    Combina título y abstract en el formato sugerido para SPECTER2.
    La columna 'abstract' en works_flat ya viene reconstruida desde el inverted index.
    """
    texts = []
    for _, row in df.iterrows():
        title = str(row.get('title', '') or '')
        # 'abstract' es la columna materializada en works_flat (reconstruida del inverted index)
        abstract = str(row.get('abstract', '') or row.get('abstract_raw', '') or '')
        text = f"Title: {title} [SEP] Abstract: {abstract}"
        texts.append(text)
    return texts
