import umap
import numpy as np

def reduce_dimensions(embeddings, n_components=5, n_neighbors=15, metric='cosine'):
    """
    Reduce la dimensionalidad de los embeddings usando UMAP para facilitar el clustering.
    768d -> 5d (o n_components).
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=42
    )
    
    projections = reducer.fit_transform(embeddings)
    return projections
