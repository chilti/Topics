"""
semantic/hdbscan_detector.py
Clustering de densidad variable con HDBSCAN.
metric='cosine': discriminativo hasta ~100d (McInnes & Healy 2018).
"""

import numpy as np
import hdbscan


def run_hdbscan(projections, min_cluster_size=50, min_samples=10,
                metric='cosine', cluster_selection_method='eom'):
    """
    Ejecuta HDBSCAN sobre proyecciones UMAP.
    Args:
        projections: Array (n, d), ej. (n, 30) tras UMAP.
        min_cluster_size: Reducir a 10-20 para sandboxes pequeños.
        metric: 'cosine' para d > 10; 'euclidean' solo para d ≤ 5.
    Returns:
        (labels, clusterer): labels=-1 indica ruido.
    """
    # Esta versión de hdbscan no soporta metric='cosine' con BallTree.
    # Solución equivalente: normalizar L2 los vectores y usar euclidean.
    # ||u-v||² = 2(1 - cosine(u,v)) para vectores unitarios → equivalente a coseno.
    if metric == 'cosine':
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        X = projections / np.where(norms > 0, norms, 1.0)
        effective_metric = 'euclidean'
    else:
        X = projections
        effective_metric = metric

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        metric=effective_metric,
        prediction_data=True,
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"   HDBSCAN: {n_clusters} clusters, {n_noise:,} ruido "
          f"({n_noise/len(labels)*100:.1f}%) [metric={metric}, min_size={min_cluster_size}]")
    return labels, clusterer
