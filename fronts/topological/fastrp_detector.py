"""
topological/fastrp_detector.py
FastRP sobre grafo heterogéneo usando igraph (CPU) o cuGraph (GPU).

Reemplaza Neo4j GDS que solo permite una proyección activa simultánea.
Unidad de análisis: ventana temporal individual, no el subcampo completo.

Red heterogénea:
  Work → Work        (citas directas)
  Work → Author      (coautoría)
  Work → Institution (afiliación)
  Work → Source      (revista)
"""

import numpy as np
import pandas as pd
import igraph as ig
from typing import Optional


def build_heterogeneous_graph(
    df_citations: pd.DataFrame,
    df_meta: pd.DataFrame,
    include_authors: bool = True,
    include_institutions: bool = True,
    include_sources: bool = True
) -> tuple:
    """
    Construye un grafo heterogéneo en igraph para los papers de una ventana.

    Args:
        df_citations: DataFrame con ['source_id', 'target_id'] — pares de citas.
        df_meta: DataFrame con ['id', 'author_ids', 'institution_ids', 'source_id']
                 de los papers del bin.
        include_*: Controla qué tipos de relaciones incluir.

    Returns:
        (graph, work_ids): grafo igraph y lista de IDs de papers (para recuperar embeddings).
    """
    work_ids = df_meta['id'].tolist()
    node_to_idx = {nid: i for i, nid in enumerate(work_ids)}
    next_idx = len(work_ids)
    edges = []

    # --- Work → Work (citas) ---
    for _, row in df_citations.iterrows():
        s, t = row['source_id'], row['target_id']
        if s in node_to_idx and t in node_to_idx:
            edges.append((node_to_idx[s], node_to_idx[t]))

    # --- Work → Author ---
    if include_authors and 'author_ids' in df_meta.columns:
        author_map = {}
        for _, row in df_meta.iterrows():
            w_idx = node_to_idx.get(row['id'])
            if w_idx is None:
                continue
            authors = row.get('author_ids', []) or []
            for aid in (authors if isinstance(authors, list) else []):
                if aid not in author_map:
                    author_map[aid] = next_idx
                    next_idx += 1
                edges.append((w_idx, author_map[aid]))

    # --- Work → Institution ---
    if include_institutions and 'institution_ids' in df_meta.columns:
        inst_map = {}
        for _, row in df_meta.iterrows():
            w_idx = node_to_idx.get(row['id'])
            if w_idx is None:
                continue
            insts = row.get('institution_ids', []) or []
            for iid in (insts if isinstance(insts, list) else []):
                if iid not in inst_map:
                    inst_map[iid] = next_idx
                    next_idx += 1
                edges.append((w_idx, inst_map[iid]))

    # --- Work → Source (revista) ---
    if include_sources and 'source_id' in df_meta.columns:
        source_map = {}
        for _, row in df_meta.iterrows():
            w_idx = node_to_idx.get(row['id'])
            sid = row.get('source_id')
            if w_idx is None or not sid:
                continue
            if sid not in source_map:
                source_map[sid] = next_idx
                next_idx += 1
            edges.append((w_idx, source_map[sid]))

    g = ig.Graph(n=next_idx, edges=edges, directed=False)
    print(f"   Grafo heterogéneo: {next_idx:,} nodos, {len(edges):,} aristas "
          f"({len(work_ids):,} papers)")
    return g, work_ids


def run_fastrp_igraph(
    graph: ig.Graph,
    n_papers: int,
    embedding_dim: int = 128,
    n_iterations: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    FastRP manual sobre igraph usando multiplicación sparse.
    Captura la posición estructural de cada nodo en el grafo heterogéneo.

    Solo retorna los embeddings de los primeros n_papers nodos (los papers).
    Los nodos de autor/institución/fuente se usan para propagación pero no se retornan.

    Args:
        graph: Grafo igraph heterogéneo.
        n_papers: Número de papers (primeros nodos en el índice).
        embedding_dim: Dimensiones del embedding FastRP.
        n_iterations: Iteraciones de propagación (L=3 es suficiente para grafos pequeños).

    Returns:
        Array (n_papers, embedding_dim) de embeddings topológicos.
    """
    np.random.seed(random_state)
    n_nodes = graph.vcount()

    # Proyección aleatoria inicial (todos los nodos)
    R = np.random.randn(n_nodes, embedding_dim).astype(np.float32) / np.sqrt(embedding_dim)

    # Matriz de adyacencia sparse
    A = np.array(graph.get_adjacency_sparse().todense(), dtype=np.float32)

    # Propagación L iteraciones: H = A^L @ R (normalizado por grado)
    degrees = np.array(graph.degree(), dtype=np.float32)
    degrees = np.where(degrees > 0, degrees, 1.0)
    D_inv = np.diag(1.0 / degrees)

    H = R.copy()
    for i in range(n_iterations):
        H = D_inv @ A @ H + R
    # Normalizar embeddings de papers
    embeddings = H[:n_papers]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)
    return embeddings


def run_fastrp_igraph_sparse(
    graph: ig.Graph,
    n_papers: int,
    embedding_dim: int = 128,
    n_iterations: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    FastRP con scipy.sparse para grafos grandes (>500K nodos).
    Evita densificar la matriz de adyacencia.
    """
    from scipy.sparse import diags
    np.random.seed(random_state)
    n_nodes = graph.vcount()

    A_sparse = graph.get_adjacency_sparse()
    A_sparse = A_sparse.astype(np.float32)

    degrees = np.array(A_sparse.sum(axis=1)).flatten()
    degrees = np.where(degrees > 0, degrees, 1.0)
    D_inv = diags(1.0 / degrees)
    A_norm = D_inv @ A_sparse

    R = np.random.randn(n_nodes, embedding_dim).astype(np.float32) / np.sqrt(embedding_dim)

    H = R.copy()
    for _ in range(n_iterations):
        H = A_norm @ H + R

    embeddings = H[:n_papers]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms > 0, norms, 1.0)
