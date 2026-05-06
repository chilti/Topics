"""
structural/leiden_detector.py
Detección de comunidades con el algoritmo de Leiden sobre la matriz de Salton.
"""

import igraph as ig
import leidenalg
import numpy as np


def run_leiden(S_matrix, resolution: float = 1.0) -> list:
    """
    Ejecuta Leiden sobre la matriz de Salton (sparse).
    """
    # Extraer aristas del triángulo superior (evitar duplicados)
    rows, cols = S_matrix.nonzero()
    upper_mask = rows < cols
    
    edges = list(zip(rows[upper_mask], cols[upper_mask]))
    weights = S_matrix.data[upper_mask]

    if len(edges) == 0:
        # print("   ⚠️ Ninguna arista supera el umbral. Bin sin estructura.")
        return list(range(S_matrix.shape[0]))

    g = ig.Graph(n=S_matrix.shape[0], edges=edges, directed=False)
    g.es['weight'] = weights.tolist()

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es['weight'],
        resolution_parameter=resolution,
        seed=42
    )

    n_clusters = len(set(partition.membership))
    print(f"   Leiden: {n_clusters} clusters (γ={resolution}, {len(edges):,} aristas)")
    return partition.membership
