"""
structural/citation_network.py
Construye la matriz de acoplamiento bibliográfico y normaliza con coseno de Salton.

Corpus abierto (OPEN_CORPUS=True): incluye referencias externas al subcampo,
capturando frentes interdisciplinares (Glänzel & Thijs, 2012).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_citation_matrix(df: pd.DataFrame, open_corpus: bool = True):
    """
    Construye la matriz de acoplamiento bibliográfico C_BC = A @ A.T.

    Args:
        df: DataFrame con columnas ['id', 'referenced_works'] para el bin temporal.
        open_corpus: Si True, incluye TODAS las referencias (internas y externas
                     al subcampo). Captura frentes interdisciplinares.
                     Si False, solo citas internas (corpus cerrado).

    Returns:
        (C_BC, work_to_idx): Matriz sparse de acoplamiento y mapa ID→índice.
    """
    unique_works = df['id'].unique()
    work_to_idx = {wid: i for i, wid in enumerate(unique_works)}
    num_works = len(unique_works)

    # Recopilar referencias según modo corpus
    all_refs = set()
    for refs in df['referenced_works']:
        if not isinstance(refs, (list, np.ndarray)):
            continue
        for r in refs:
            if r and isinstance(r, str):
                if open_corpus or r in work_to_idx:
                    all_refs.add(r)

    ref_to_idx = {rid: i for i, rid in enumerate(all_refs)}
    num_refs = len(all_refs)

    if num_refs == 0:
        return None, work_to_idx

    # Construir matriz dispersa A (papers × referencias)
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        refs = row.get('referenced_works', [])
        if not isinstance(refs, (list, np.ndarray)):
            continue
        w_idx = work_to_idx[row['id']]
        for ref in refs:
            if ref in ref_to_idx:
                rows.append(w_idx)
                cols.append(ref_to_idx[ref])
                data.append(1)

    A = csr_matrix((data, (rows, cols)), shape=(num_works, num_refs), dtype=np.float32)

    # C_BC = A @ A.T: número de referencias compartidas entre pares de papers
    C_BC = A @ A.T

    corpus_mode = "abierto" if open_corpus else "cerrado"
    print(f"   C_BC: {num_works} papers × {num_refs} refs ({corpus_mode}), "
          f"densidad={len(data)/(num_works*num_refs):.4%}")

    return C_BC, work_to_idx


def normalize_salton(C) -> np.ndarray:
    """
    Aplica normalización coseno de Salton: S_ik = C_ik / sqrt(k_i * k_k).
    Neutraliza el sesgo por tamaño de bibliografía.

    Returns: matriz densa S (valores en [0, 1]).
    """
    C_dense = C.toarray().astype(np.float64)
    k = np.sqrt(np.diag(C_dense))
    k_inv = np.where(k > 0, 1.0 / k, 0.0)
    S = C_dense * k_inv[:, np.newaxis] * k_inv[np.newaxis, :]
    return S


def apply_salton_threshold(S: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Filtra aristas débiles aplicando umbral sobre el coseno de Salton.
    threshold=0.1 es el estándar de facto (Waltman, 2016) — scale-free.

    Returns: S con valores < threshold puestos a 0.
    """
    S_filtered = np.where(S >= threshold, S, 0.0)
    np.fill_diagonal(S_filtered, 0.0)
    n_edges = np.count_nonzero(S_filtered) // 2
    print(f"   Aristas tras umbral Salton ≥ {threshold}: {n_edges:,}")
    return S_filtered
