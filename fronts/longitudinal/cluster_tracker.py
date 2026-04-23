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
    """
    bins = sorted(df[bin_col].unique())
    transitions = []
    
    for i in range(len(bins) - 1):
        bin_curr = bins[i]
        bin_next = bins[i+1]
        
        clusters_curr = df[df[bin_col] == bin_curr][cluster_col].unique()
        clusters_next = df[df[bin_col] == bin_next][cluster_col].unique()
        
        # Ignorar ruido (-1 en HDBSCAN)
        clusters_curr = [c for c in clusters_curr if c != -1]
        clusters_next = [c for c in clusters_next if c != -1]
        
        for c1 in clusters_curr:
            set1 = set(df[(df[bin_col] == bin_curr) & (df[cluster_col] == c1)]['id'])
            for c2 in clusters_next:
                set2 = set(df[(df[bin_col] == bin_next) & (df[cluster_col] == c2)]['id'])
                
                sim = calculate_jaccard(set1, set2)
                if sim > 0.05: # Umbral bajo para capturar todas las conexiones
                    transitions.append({
                        'from_bin': bin_curr,
                        'to_bin': bin_next,
                        'from_cluster': c1,
                        'to_cluster': c2,
                        'jaccard': sim,
                        'shared_docs': len(set1.intersection(set2))
                    })
                    
    return pd.DataFrame(transitions)
