import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

def extract_cluster_terms(
    df_cluster: pd.DataFrame, 
    text_col: str = 'abstract', 
    top_n: int = 10,
    max_features: int = 1000
) -> List[str]:
    """
    Extrae los términos más representativos de un clúster usando TF-IDF.
    """
    texts = df_cluster[text_col].dropna().tolist()
    if not texts:
        return []

    # Usamos stop_words='english' ya que la mayoría de la literatura científica es en inglés
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=max_features,
        ngram_range=(1, 2)  # Unigramas y bigramas
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        # Sumamos los pesos de cada término en todo el clúster
        sums = tfidf_matrix.sum(axis=0)
        data = []
        for col, term in enumerate(vectorizer.get_feature_names_out()):
            data.append((term, sums[0, col]))
            
        ranking = pd.DataFrame(data, columns=['term', 'rank'])
        ranking = ranking.sort_values('rank', ascending=False)
        
        return ranking['term'].head(top_n).tolist()
    except Exception as e:
        print(f"      [TF-IDF] Error: {e}")
        return []

def get_top_titles(df_cluster: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Retorna los títulos de los papers más citados del clúster (como referencia para el LLM).
    """
    if 'cited_by_count' in df_cluster.columns:
        top_papers = df_cluster.sort_values('cited_by_count', ascending=False).head(top_n)
    else:
        top_papers = df_cluster.head(top_n)
        
    return top_papers['title'].tolist()
