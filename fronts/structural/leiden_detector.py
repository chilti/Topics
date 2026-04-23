import igraph as ig
import leidenalg
import numpy as np

def run_leiden(S_matrix, resolution=1.0):
    """
    Ejecuta el algoritmo de Leiden sobre una matriz de adyacencia pesada.
    """
    # 1. Crear grafo desde matriz de adyacencia (simétrica)
    # Ignorar la diagonal (auto-referencias)
    np.fill_diagonal(S_matrix, 0)
    
    # Solo pesos positivos
    rows, cols = np.where(S_matrix > 0)
    edges = list(zip(rows, cols))
    weights = S_matrix[rows, cols]
    
    # Como la matriz es simétrica, nos quedamos solo con la parte superior para igraph
    upper_mask = rows < cols
    edges = [edges[i] for i in range(len(edges)) if upper_mask[i]]
    weights = weights[upper_mask]
    
    g = ig.Graph(n=S_matrix.shape[0], edges=edges, directed=False)
    g.es['weight'] = weights
    
    # 2. Ejecutar Leiden (Modularidad RBConfiguration)
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition, 
        weights=g.es['weight'],
        resolution_parameter=resolution
    )
    
    return partition.membership
