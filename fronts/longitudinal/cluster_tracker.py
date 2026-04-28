import pandas as pd
import numpy as np

def calculate_jaccard(set_a, set_b):
    """Calcula el índice de similitud de Jaccard entre dos conjuntos."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def track_clusters(df, cluster_col, bin_col):
    """
    Rastrea la evolución de los clusters a través de los bins temporales.
    Retorna un grafo de transiciones.
    Pre-calcula los conjuntos de IDs por cluster para evitar O(clusters² × N) reconstrucciones.
    """
    bins = sorted(df[bin_col].unique())
    transitions = []
    
    # Pre-calcular todos los conjuntos una sola vez
    cluster_sets = {}
    for b in bins:
        df_bin = df[df[bin_col] == b]
        for c in df_bin[cluster_col].unique():
            if c != -1:  # Ignorar ruido HDBSCAN
                cluster_sets[(b, c)] = set(df_bin[df_bin[cluster_col] == c]['id'])
    
    for i in range(len(bins) - 1):
        bin_curr = bins[i]
        bin_next = bins[i + 1]
        
        clusters_curr = [c for c in df[df[bin_col] == bin_curr][cluster_col].unique() if c != -1]
        clusters_next = [c for c in df[df[bin_col] == bin_next][cluster_col].unique() if c != -1]
        
        for c1 in clusters_curr:
            set1 = cluster_sets.get((bin_curr, c1), set())
            for c2 in clusters_next:
                set2 = cluster_sets.get((bin_next, c2), set())
                
                sim = calculate_jaccard(set1, set2)
                if sim > 0.05:  # Umbral bajo para capturar todas las conexiones
                    transitions.append({
                        'from_bin': bin_curr,
                        'to_bin': bin_next,
                        'from_cluster': c1,
                        'to_cluster': c2,
                        'jaccard': sim,
                        'shared_docs': len(set1.intersection(set2))
                    })
                    
    return pd.DataFrame(transitions)
