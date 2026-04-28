"""
structural/leiden_detector.py
Detección de comunidades con el algoritmo de Leiden sobre la matriz de Salton.
"""

import igraph as ig
import leidenalg
import numpy as np


def run_leiden(S_matrix: np.ndarray, resolution: float = 1.0) -> list:
    """
    Ejecuta Leiden sobre la matriz de Salton ya filtrada (umbral aplicado previamente).

    Args:
        S_matrix: Matriz densa simétrica (valores en [0,1], diagonal=0,
                  aristas débiles ya eliminadas con apply_salton_threshold).
        resolution: Parámetro γ de RBConfigurationVertexPartition.
                    Valores > 1.0 → más clusters pequeños.
                    Valores < 1.0 → clusters más grandes.

    Returns:
        Lista de enteros con la asignación de cluster por nodo.
    """
    np.fill_diagonal(S_matrix, 0)

    # Extraer aristas del triángulo superior (evitar duplicados en igraph)
    rows, cols = np.where(S_matrix > 0)
    upper_mask = rows < cols
    edges = list(zip(rows[upper_mask], cols[upper_mask]))
    weights = S_matrix[rows[upper_mask], cols[upper_mask]]

    if len(edges) == 0:
        print("   ⚠️  Ninguna arista supera el umbral de Salton. Bin sin estructura.")
        return list(range(S_matrix.shape[0]))  # Cada paper en su propio cluster

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
