import hdbscan
import numpy as np

def run_hdbscan(projections, min_cluster_size=30, min_samples=5):
    """
    Ejecuta HDBSCAN sobre las proyecciones UMAP para detectar clusters de densidad variable.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom', # Excess of Mass (más estable)
        prediction_data=True
    )
    
    labels = clusterer.fit_predict(projections)
    return labels, clusterer
