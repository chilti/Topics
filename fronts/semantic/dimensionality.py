"""
semantic/dimensionality.py
Reducción dimensional con UMAP para preprocesar embeddings antes de HDBSCAN.

Pipeline recomendado: UMAP(768→30d, coseno) → HDBSCAN(coseno)
- UMAP no lineal preserva la geometría de embeddings de transformers mejor que PCA.
- Métrica coseno: discriminativa para embeddings de texto en alta dimensionalidad.
- n_components=30: equilibrio entre preservación de estructura y velocidad HDBSCAN.
"""

import numpy as np
import umap


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 30,
    n_neighbors: int = 30,
    metric: str = 'cosine',
    min_dist: float = 0.0,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce la dimensionalidad de embeddings usando UMAP.

    Args:
        embeddings: Array (n_papers, embed_dim), ej. (n, 768) para SPECTER2.
        n_components: Dimensiones de salida. 30 para HDBSCAN; 2 para visualización.
        n_neighbors: Vecinos locales. 30 para subcampos grandes.
        metric: 'cosine' para embeddings de transformers (por defecto).
        min_dist: 0.0 maximiza la densidad local, óptimo para clustering.
        random_state: Semilla para reproducibilidad.

    Returns:
        Array (n_papers, n_components) de proyecciones.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        random_state=random_state,
        low_memory=False  # Más rápido cuando hay RAM disponible
    )
    projections = reducer.fit_transform(embeddings)
    print(f"   UMAP: {embeddings.shape[1]}d → {n_components}d "
          f"(metric={metric}, n_neighbors={n_neighbors})")
    return projections


def reduce_for_visualization(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Reducción a 2D para visualización (Plotly scatter). No usar para clustering.
    """
    return reduce_dimensions(
        embeddings, n_components=2, n_neighbors=30,
        metric='cosine', min_dist=0.1, random_state=random_state
    )
